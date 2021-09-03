using MPI
using PPMD
using Test
using LinearAlgebra



@testset "PBC apply $spec" for spec in (KACPU(), KACUDADevice())

    target_device = spec

    N = 10

    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(3, position=true),
             "B" => ParticleDat(3),
             "A" => ParticleDat(1),
        ),
        target_device
    )
    

    pinitial = rand_within_extents(N, domain.extent)

    pmodified = copy(pinitial)
    for dx in 1:3
        pmodified[:, dx] += rand(-10:10, (N, 1)) * extents[dx]
    end

    add_particles(
        A,
        Dict(
             "P" => pmodified,
             "A" => rand(Float64, (N, 1))
        )
    )

    loop = get_boundary_condition_loop(A.domain.boundary_condition, A)

    execute(loop)

    @test norm(A["P"][:, :] .- pinitial[:, :], Inf) < 1E-14

end
