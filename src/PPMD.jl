module PPMD
using MPI

export ParticleLoop, KernelAbstractionsDevice, KACPU, KACUDADevice

include("profiling.jl")

function __init__()
    if !(MPI.Initialized())
        #MPI.Init(finalize_atexit=true)
        MPI.Init_thread(MPI.THREAD_FUNNELED, finalize_atexit=true)
    end
    @assert MPI.Query_thread() in [MPI.THREAD_FUNNELED, MPI.THREAD_SERIALIZED, MPI.THREAD_MULTIPLE]
    #@assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
    

    # Set MPI vars in profile
    reset_profile()
end


# Try to determine the local rank on each node
function get_local_rank()
    
    # OpenMPI
    if ((local_rank=parse(Int64, get(ENV, "OMPI_COMM_WORLD_LOCAL_RANK", "-1"))) > -1) 
        return local_rank
    end
    
    # MPICH2
    if ((local_rank=parse(Int64, get(ENV, "MV2_COMM_WORLD_LOCAL_RANK", "-1"))) > -1) 
        return local_rank
    end

    # IBM? also set (sometimes?) by Intel MPI
    if ((local_rank=parse(Int64, get(ENV, "MPI_LOCALRANKID", "-1"))) > -1) 
        return local_rank
    end
    
    return 0
end

LOCAL_RANK = get_local_rank()

include("mpi_utility.jl")
include("hash.jl")
include("access.jl")
include("kernel.jl")
include("loop_execution.jl")
include("target_devices.jl")
include("neighbour_transfer.jl")
include("particle_group.jl")
include("particle_dat.jl")
include("global_array.jl")
include("particle_loop.jl")
include("domain.jl")
include("utility.jl")
include("mesh.jl")
include("cell_dat.jl")
include("cell_to_particle_map.jl")
include("pair_loop.jl")
include("output.jl")


end # module
