using MPI
using PPMD
using Test
using LinearAlgebra
using Random

@testset "PBC global move $spec" for spec in (KACPU(), KACUDADevice())

    target_device = spec

    N = 20

    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(3, position=true),
             "STARTING_RANK" => ParticleDat(1, Int64),
             "IDA" => ParticleDat(1, Int64),
             "IDB" => ParticleDat(1, Int64),
        ),
        target_device
    )
    
    size = MPI.Comm_size(domain.comm)
    rank = MPI.Comm_rank(domain.comm)
    
    rng = MersenneTwister(1234)
    dest_ranks = rand(rng, (0:size-1), N, 1)

    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
             "_owning_rank" => dest_ranks,
             "STARTING_RANK" => rank * ones(Int64, (N, 1)),
             "IDA" => reshape([ix for ix in 1:N], (N, 1)) .+ N*rank,
             "IDB" => reshape([ix for ix in 1:N], (N, 1)) .+ N*rank,
        )
    )
    
    global_transfer_to_rank(A)

    npart_total = MPI.Allreduce([A.npart_local], MPI.SUM, domain.comm)
    @test npart_total[1] == size * N
    @test norm(A["_owning_rank"][:, 1] .- rank, Inf) < 1E-15

    for ix in 1:A.npart_local
        @test norm(A["IDA"][:, 1] - A["IDB"][:, 1], Inf) == 0
    end


end
