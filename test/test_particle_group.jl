using PPMD
using Test
using LinearAlgebra


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
    
    N1 = 5

    add_particles(
        A,
        Dict(
             "P" => Pi[1:N1, :],
             "B" => Bi[1:N1, :],
             "A" => Ai[1:N1, :],
        )
    )
    
    # TODO change for MPI parallelism
    @test A.npart_local == N1
    for dx in ("P", "B", "A")
        @test A[dx].npart_local == N1
    end
    
    add_particles(
        A,
        Dict(
             "P" => Pi[N1 + 1:N, :],
             "B" => Bi[N1 + 1:N, :],
             "A" => Ai[N1 + 1:N, :],
        )
    )

    # TODO change for MPI parallelism
    @test A.npart_local == N
    for dx in ("P", "B", "A")
        @test A[dx].npart_local == N
    end


    for px in 1:N
        orig_index = A["A"][px, 1]
        @test norm(A["P"][px, :] - Pi[orig_index, :], Inf) <= 1E-15
        @test norm(A["B"][px, :] - Bi[orig_index, :], Inf) <= 1E-15
    end

    
    rm_count = 0
    to_remove = (1, 3, 8)
    for rx in to_remove
        
        remove_particles(A, (rx,))
        rm_count += 1
        # TODO change for MPI parallelism
        @test A.npart_local == N - rm_count
        N_remain = N - rm_count
        for dx in ("P", "B", "A")
            @test A[dx].npart_local == N_remain
        end       

        for px in 1:N_remain
            orig_index = A["A"][px, 1]
            @test norm(A["P"][px, :] - Pi[orig_index, :], Inf) <= 1E-15
            @test norm(A["B"][px, :] - Bi[orig_index, :], Inf) <= 1E-15
        end

    end


end
















