using PPMD
using Test

@testset "particle group" begin

    target_device = KACPU()

    N = 10

    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(3, position=true),
             "B" => ParticleDat(3),
             "A" => ParticleDat(1, Int64),
        ),
        target_device
    )

    @test A.domain == domain
    
    @test A["P"].particle_group == A
    @test A["B"].particle_group == A
    @test A["A"].particle_group == A

    @test A.position_dat == "P"
    @test A.npart_local == 0
    @test A.compute_target == target_device

    @test A["P"].compute_target == target_device
    @test A["B"].compute_target == target_device
    @test A["A"].compute_target == target_device


    N = 10
    Pi = rand_within_extents(N, domain.extent)
    Bi = rand(Float64, (N, 3))
    Ai = reshape(range(1, N, step=1), (N, 1))

    add_particles(
        A,
        Dict(
             "P" => Pi,
             "B" => Bi,
             "A" => Ai,
        )
    )
    
    for px in 1:N
        println(A["P"][px, 1])
    end




end
