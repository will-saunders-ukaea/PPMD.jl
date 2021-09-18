module PPMD
using MPI

export ParticleLoop, KernelAbstractionsDevice, KACPU, KACUDADevice

function __init__()
    if !(MPI.Initialized())
        MPI.Init(finalize_atexit=true)
    end
end

include("access.jl")
include("kernel.jl")
include("loop_execution.jl")
include("target_devices.jl")
include("particle_group.jl")
include("particle_dat.jl")
include("global_array.jl")
include("particle_loop.jl")
include("domain.jl")
include("utility.jl")



end # module
