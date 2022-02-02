using PPMD
using Test
using LinearAlgebra
using Random
using MPI

@testset "particle group $target_device" for target_device in (KACPU(), KACUDADevice())

    N = 10

    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    comm = domain.comm
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

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



    rng = MersenneTwister(1234)


    N = 100
    Pi = rand_within_extents(N, domain.extent, rng)
    Bi = rand(rng, Float64, (N, 3))
    Ai = reshape(range(1, N, step=1), (N, 1))
    
    N1 = 50

    add_particles(
        A,
        Dict(
             "P" => Pi[1:N1, :],
             "B" => Bi[1:N1, :],
             "A" => Ai[1:N1, :],
        )
    )
 
    npart_total = MPI.Allreduce(A.npart_local, MPI.SUM, comm)
    @assert npart_total == N1 * size

    add_particles(
        A,
        Dict(
             "P" => Pi[N1 + 1:N, :],
             "B" => Bi[N1 + 1:N, :],
             "A" => Ai[N1 + 1:N, :],
        )
    )

    npart_total = MPI.Allreduce(A.npart_local, MPI.SUM, comm)
    @assert npart_total == N1 * 2 * size

    add_particles(
        A,
        Dict(
             "P" => Pi[N1 + 1:N, :],
             "B" => Bi[N1 + 1:N, :],
             "A" => Ai[N1 + 1:N, :],
        )
    )

    npart_total = MPI.Allreduce(A.npart_local, MPI.SUM, comm)
    @assert npart_total == N1 * 3 * size


    npart_local = A.npart_local
    for dx in ("P", "B", "A")
        @test A[dx].npart_local == npart_local
    end


    for px in 1:A.npart_local
        orig_index = A["A"][px, 1]
        @test norm(A["P"][px, :] - Pi[orig_index, :], Inf) <= 1E-15
        @test norm(A["B"][px, :] - Bi[orig_index, :], Inf) <= 1E-15
    end
    

    npart_local = A.npart_local
    npart_left = npart_local

    remove_particles(A, [])
    @test A.npart_local == npart_local

    while (npart_left > 0)
        
        npart_to_remove = rand(rng, 1:npart_left)
        to_remove = Set(rand(rng, 1:npart_left, npart_to_remove))
        npart_to_remove = length(to_remove)

        remove_particles(A, to_remove)

        npart_left -= npart_to_remove
        @assert A.npart_local == npart_left

        for dx in ("P", "B", "A")
            @test A[dx].npart_local == npart_left
        end       

        for px in 1:npart_left
            orig_index = A["A"][px, 1]
            @test norm(A["P"][px, :] - Pi[orig_index, :], Inf) <= 1E-15
            @test norm(A["B"][px, :] - Bi[orig_index, :], Inf) <= 1E-15
        end


    end

    @assert A.npart_local == 0

    free(A)
end
