

using KernelAbstractions
using FunctionWrappers: FunctionWrapper


function ParticleLoop(
    target,
    kernel,
    args
)
    # Assemble the kernel function args
    kernel_args = join(keys(args), ',')
    
    # TODO more complicated access, e.g. reductions should be 
    # constructed here
    
    # Assemble the kernel function
    kernel_func = """
    @kernel function kernel_wapper($kernel_args)
        ix = @index(Global)
        
        $(kernel.source)

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


