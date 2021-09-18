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
        return "global_$kernel_sym"
    end
end




function ParticleLoop(
    target,
    kernel,
    args
)
    # ensure args have a consistent ordering
    args = make_args_sorted(args)

    # Assemble the kernel function parameters
    kernel_params = join([get_wrapper_param(px.first, px.second[1], px.second[2], target) for px in args], ',')
    

    pre_kernel_launch = join([get_wrapper_param(px.first, px.second[1], px.second[2], target) for px in args], "\n")


    post_kernel_launch = ""
    
    

    # Assemble the kernel function
    kernel_func = """
    @kernel function kernel_wapper($kernel_params)
        _local_ix = @index(Local)
        _group_ix = @index(Group)
        ix = @index(Global)
        
        $pre_kernel_launch
        
        @inbounds begin
        $(kernel.source)
        end

        $post_kernel_launch

    end
    """

    println(kernel_func)

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
        # TODO need a robust way to handle temporaries from reductions etc
        call_args = (ax[1].data for ax in values(args))

        event = loop_func(call_args..., ndrange=N)

        # alternatively this could return the event to such that multiple
        # kernels can be launched in parallel?
        wait(event)   
    end
    
    l.execute = loop_wrapper

    return l

end


