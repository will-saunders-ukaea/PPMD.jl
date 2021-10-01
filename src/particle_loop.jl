using KernelAbstractions
using FunctionWrappers: FunctionWrapper
using DataStructures


function flatten(c)
    return collect(Iterators.flatten(c))
end


function flatten_join(c, delimiter)
    return join(
        flatten(c),
        delimiter
    )
end


function make_args_sorted(args)
    d = SortedDict{String, Tuple}(args)
    return d
end


"""
Construct the parameters for the kernel function
"""
function get_wrapper_param(kernel_sym, dat::ParticleDat, access_mode, target)
    return (kernel_sym,)
end
function get_wrapper_param(kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return (kernel_sym,)
    end

    if (access_mode == INC)
        return ("_global_$kernel_sym",)
    end
end


"""
Generate code for the data structure and access type prior to the kernel launch 
"""
function get_pre_kernel_launch(kernel_sym, dat::ParticleDat, access_mode, target)
    return ""
end
function get_pre_kernel_launch(kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return ""
    end

    req_size = dat.ncomp * target.workgroup_size

    return """
    $kernel_sym = @private $(dat.dtype) ($(dat.ncomp),)
    for _rx in 1:$(dat.ncomp)
        @inbounds $kernel_sym[_rx] = 0
    end
    _reduce_$kernel_sym = @localmem $(dat.dtype) $req_size
    if (_local_ix == 1)
        for _rx in 1:$req_size
            @inbounds _reduce_$kernel_sym[_rx] = 0
        end
    end
    """
end


"""
Generate code for the data structure and access post kernel launch
"""
function get_post_kernel_launch(kernel_sym, dat::ParticleDat, access_mode, target)
    return ""
end
function get_post_kernel_launch(kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return ""
    end
    ncomp = dat.ncomp
    a = """
    @inbounds begin
        
        for _rx in 1:$ncomp
            _reduce_$(kernel_sym)[(_local_ix-1) * $ncomp + _rx] = $kernel_sym[_rx]  
        end
        @synchronize
        # TODO make this a tree reduction
        if (_local_ix == 1)
            for _bx in 2:$(target.workgroup_size)
                for _rx in 1:$ncomp
                    _reduce_$(kernel_sym)[_rx] += _reduce_$(kernel_sym)[(_bx-1) * $ncomp + _rx]
                end
            end
            
            # write to the global data for this workgroup
            # TODO investigate atomics for this
            for _rx in 1:$ncomp
                _global_$kernel_sym[(_group_ix - 1) * $ncomp + _rx] = _reduce_$kernel_sym[_rx]
            end
        end
    end
    """
    return a
end


"""
Establish if this data structure and access combination require a sync before
the kernel is launched.
"""
function get_pre_kernel_sync(kernel_sym, dat::ParticleDat, access_mode, target)
    return false
end
function get_pre_kernel_sync(kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return false
    else
        return true
    end
end


"""
Get the calling arguments for the data structure and access mode. This may 
also handle any communication required before the loop launches. e.g.
halo exchanges.
"""
function get_loop_args(N, kernel_sym, dat::ParticleDat, access_mode, target)
    return (dat.data,)
end
function get_loop_args(N, kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return (dat.data,)
    end
    if (access_mode == INC)
        ngroups = Int(ceil(N / target.workgroup_size))
    end
    return (target.ArrayType{dat.dtype}(undef, (dat.ncomp, ngroups)),)
end


"""
After a loop has completed perform actions for each data structure and access
type, e.g. reduction operations.
"""
function post_loop(N, kernel_sym, dat::ParticleDat, access_mode, target, arg)
    return
end
function post_loop(N, kernel_sym, dat::GlobalArray, access_mode, target, arg)
    if (!access_mode.write)
        return dat.data
    end
    
    arg = sum(arg, dims=2)
    host_data = get_array_on_host(arg)
    reduce_result = MPI.Allreduce(view(host_data, 1:dat.ncomp),  MPI.SUM, dat.comm)
    increment(dat, reduce_result)
end


"""
Julia does not seem to natively support atomic
    
    old = a[i]
    a[i] += b

at the moment (CUDA.jl does with "old = CUDA.atomic_add!(pointer(a, i), b)).
Here we construct an atomic function for use in kernels that fills the gap on
the host. The kernel should be written with

    old = ATOMIC_ADD(a, ix, b)

to indicate the atomic add.
"""
function cuda_atomic_add(a, ix, b)
    return CUDA.atomic_add!(pointer(a, ix), b)
end
function cpu_atomic_add(a::Array{Int64}, ix::Int64, b::Int64)
    old = Base.Threads.llvmcall(
        "%ptr = inttoptr i64 %0 to i64*\n%rv = atomicrmw add i64* %ptr, i64 %1 acq_rel\nret i64 %rv\n",
        Int64, Tuple{Ptr{Int64},Int64}, pointer(a, ix), b
    )
    return old

    #julia_type = eltype(a)
    #@assert (julia_type <: Int64) || (julia_type <: Float64)
    #
    #if julia_type == Int64
    #    old = Base.Threads.llvmcall(
    #        "%ptr = inttoptr i64 %0 to i64*\n%rv = atomicrmw add i64* %ptr, i64 %1 acq_rel\nret i64 %rv\n",
    #        Int64, Tuple{Ptr{Int64},Int64}, pointer(a, ix), b
    #    )
    #else 
    #    old = Base.Threads.llvmcall(
    #        "%ptr = inttoptr i64 %0 to i64*\n%rv = atomicrmw add i64* %ptr, f64 %1 acq_rel\nret f64 %rv\n",
    #        Float64, Tuple{Ptr{Float64}, Float64}, pointer(a, ix), b
    #    )
    #end

    #return old
end
function get_atomic_add(target::KACUDADevice)
    return "cuda_atomic_add"
end
function get_atomic_add(target::KACPU)
    @assert sizeof(Cptrdiff_t) == 8
    return "cpu_atomic_add"
end

function ParticleLoop(
    target,
    kernel,
    args
)
    # ensure args have a consistent ordering
    args = make_args_sorted(args)

    # Assemble the kernel function parameters
    kernel_params = flatten_join(
        [get_wrapper_param(px.first, px.second[1], px.second[2], target) for px in args],
        ','
    )
    pre_kernel_launch = join(
        [get_pre_kernel_launch(px.first, px.second[1], px.second[2], target) for px in args], 
        "\n"
    )
    post_kernel_launch = join(
        [get_post_kernel_launch(px.first, px.second[1], px.second[2], target) for px in args], 
        "\n"
    )
    
    # is a syncronize required before the kernel call? e.g. for local_mem init
    if Bool(sum([get_pre_kernel_sync(px.first, px.second[1], px.second[2], target) for px in args]))
        pre_kernel_sync = "@synchronize"
    else
        pre_kernel_sync = ""
    end


    # Assemble the kernel function
    kernel_func = """
    @kernel function kernel_wapper($kernel_params)
        ATOMIC_ADD = $(get_atomic_add(target))

        _local_ix = @index(Local)
        _group_ix = @index(Group)
        ix = @index(Global)
        
        $pre_kernel_launch
        $pre_kernel_sync

        @inbounds begin
        $(kernel.source)
        end

        $post_kernel_launch

    end
    """

    l = Task(
        kernel.name * "_" * string(target) * "_ParticleLoop",
        () -> return
    )

    # Pass the kernel function to KernelAbstractions
    @eval new_kernel() = $(Meta.parse(kernel_func))
    
    ka_methods = Base.invokelatest(new_kernel)
    loop_func = Base.invokelatest(ka_methods, target.device, target.workgroup_size)

    # Create the function which can be passed to execute
    function loop_wrapper()
        # call the loop itself
        
        # Find the number of local particles
        N = -1
        for datx in values(args)
            if (typeof(datx[1]) <: ParticleDat)
                N = datx[1].npart_local
                break
            end
        end
        @assert N > -1

        # Assemble the args for the call.
        call_args = flatten([get_loop_args(N, px.first, px.second[1], px.second[2], target) for px in args])

        event = loop_func(call_args..., ndrange=N)
 
        # alternatively this could return the event to such that multiple
        # kernels can be launched in parallel?
        wait(event)

        # handle any post loop procedures
        for px in zip(args, call_args)
            post_loop(N, px[1].first, px[1].second[1], px[1].second[2], target, px[2])
        end
   
    end
    
    l.execute = loop_wrapper

    return l

end


