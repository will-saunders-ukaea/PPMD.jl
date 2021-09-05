using MPI
using PPMD
using Test
using LinearAlgebra


@testset "PBC global move $spec" for spec in (KACPU(), KACUDADevice())

    target_device = spec

    N = 100

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

    global_move(A)

    display(A["_owning_rank"][:, :])


end
