

using KernelAbstractions



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
    @kernel function foo($kernel_args)
        ix = @index(Global)
        
        $(kernel.source)

    end
    """
    
    # Pass the kernel function to KernelAbstractions
    @eval new_kernel() = $(Meta.parse(kernel_func))
    
    # Create the function which can be passed to execute
    # maybe this should be a struct that contains all the data and a function?
    function loop_wrapper()
        # TODO establish a robust method to hande the number of particles
        N = first(values(args))[1].npart_local

        # Assemble the args for the call.
        # TODO need a robust way to handle temporaries from reductions etc
        call_args = (ax[1].data for ax in values(args))
        
        # call the loop itself
        loop_func = new_kernel()(target.device, target.workgroup_size)
        event = loop_func(call_args..., ndrange=N)

        # alternatively this could return the event to such that multiple
        # kernels can be launched in parallel?
        wait(event)   
    end
    
    name = kernel.name * "_" * string(target) * "_ParticleLoop"

    l = Task(
        name,
        loop_wrapper
    )

    return l

end


