module PPMD

export ParticleLoop, execute, READ, WRITE, KernelAbstractionsDevice, KACPU, KACUDADevice


include("target_devices.jl")
include("particle_loop.jl")
include("access.jl")


function execute(loop)
    return loop()
end



end # module
