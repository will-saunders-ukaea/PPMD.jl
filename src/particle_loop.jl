using KernelAbstractions
using FunctionWrappers: FunctionWrapper
using DataStructures


function make_args_sorted(args)
    d = SortedDict{String, Tuple}(args)
    return d
end


function get_wrapper_param(kernel_sym, dat::ParticleDat, access_mode, target)
    return kernel_sym
end

function get_wrapper_param(kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return kernel_sym
    end

    if (access_mode == INC)
        return "_global_$kernel_sym"
    end
end


function get_pre_kernel_launch(kernel_sym, dat::ParticleDat, access_mode, target)
    return ""
end


function get_pre_kernel_launch(kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return ""
    end

    req_size = dat.ncomp * target.workgroup_size
    return """
    @private $kernel_sym = zeros($(dat.dtype), ($(dat.ncomp),))
    _reduce_$kernel_sym = @localmem $(dat.dtype) $req_size
        if (_local_ix == 1)
            for _rx in 1:$req_size
                @inbounds _reduce_$kernel_sym[_rx] = 0
            end
        end
    """
end


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
            for _rx in 1:$ncomp
                _global_$kernel_sym[(_group_ix - 1) * $ncomp + _rx] = _reduce_$kernel_sym[_rx]
            end
        end
    end
    """
    return a
end


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


function get_loop_args(N, kernel_sym, dat::ParticleDat, access_mode, target)
    return dat.data
end


function get_loop_args(N, kernel_sym, dat::GlobalArray, access_mode, target)
    if (!access_mode.write)
        return data.data
    end
    if (access_mode == INC)
        ngroups = Int(ceil(N / target.workgroup_size))
    end
    return target.ArrayType{dat.dtype}(undef, (dat.ncomp, ngroups))
end


function post_loop(N, kernel_sym, dat::ParticleDat, access_mode, target, arg)
    return
end


function post_loop(N, kernel_sym, dat::GlobalArray, access_mode, target, arg)
    if (!access_mode.write)
        return data.data
    end

    println(arg[:,:])
end


function ParticleLoop(
    target,
    kernel,
    args
)
    # ensure args have a consistent ordering
    args = make_args_sorted(args)

    # Assemble the kernel function parameters
    kernel_params = join(
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
    # maybe this should be a struct that contains all the data and a function?
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
        call_args = [get_loop_args(N, px.first, px.second[1], px.second[2], target) for px in args]

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


