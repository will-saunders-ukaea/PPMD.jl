using MPI
using PPMD
using Test
using LinearAlgebra



struct AA
    boundary_condition_task
end


@testset "PBC global add $spec" for spec in (KACPU(), KACUDADevice())

    target_device = spec

    N = 1

    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(3, position=true),
             "B" => ParticleDat(3),
        ),
        target_device
    )
    

    pinitial = rand_within_extents(N, domain.extent)
    ptest = rand_within_extents(N, domain.extent)


    add_particles(
        A,
        Dict(
             "P" => pinitial,
             "B" => ptest
        )
    )
    
    #initialise_particle_group_move(A)
    
    global_move(A)

    #execute(A.boundary_condition_task)
    
    #aa = AA(get_boundary_condition_loop(A.domain.boundary_condition, A))


end
