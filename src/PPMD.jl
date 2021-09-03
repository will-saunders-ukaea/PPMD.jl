module PPMD
using MPI

export ParticleLoop, READ, WRITE, KernelAbstractionsDevice, KACPU, KACUDADevice

function __init__()
    if !(MPI.Initialized())
        MPI.Init(finalize_atexit=true)
    end
end

include("kernel.jl")
include("loop_execution.jl")
include("target_devices.jl")
include("particle_loop.jl")
include("access.jl")
include("domain.jl")
include("particle_group.jl")
include("particle_dat.jl")
include("utility.jl")



end # module
