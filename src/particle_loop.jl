

using KernelAbstractions




function ParticleLoop(
    target,
    kernel,
    args
)

    

    kernel_args = join(keys(args), ',')


    kernel_func = """
    
    @kernel function foo($kernel_args)
        ix = @index(Global)
        
        $kernel

    end
    """

    println(kernel_func)

    @eval new_kernel() = $(Meta.parse(kernel_func))

    function loop_wrapper()
        N = size(first(values(args))[1], 1)
        call_args = (ax[1] for ax in values(args))

        loop_func = new_kernel()(CPU(), 8)
        event = loop_func(call_args..., ndrange=N)
        wait(event)   
    end

    return loop_wrapper

end


