module PPMD
using MPI

export ParticleLoop, KernelAbstractionsDevice, KACPU, KACUDADevice

function __init__()
    if !(MPI.Initialized())
        MPI.Init(finalize_atexit=true)
    end
end


# Try to determine the local rank on each node
function get_local_rank()

    if ((local_rank=parse(Int64, get(ENV, "OMPI_COMM_WORLD_LOCAL_RANK", "-1"))) > -1) 
        return local_rank
    end

    if ((local_rank=parse(Int64, get(ENV, "MV2_COMM_WORLD_LOCAL_RANK", "-1"))) > -1) 
        return local_rank
    end
    
    return 0
end

LOCAL_RANK = get_local_rank()


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

include("mesh.jl")
include("cell_dat.jl")


end # module
