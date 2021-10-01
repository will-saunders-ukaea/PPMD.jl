using MPI
using PPMD
using Test
using LinearAlgebra


@testset "PBC global add $spec" for spec in (KACPU(), KACUDADevice())

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
    
    comm = domain.comm
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    
    # randomly add particles on each rank that will need sending to the 
    # correct owning rank
    add_particles(
        A,
        Dict(
             "P" => pinitial,
             "B" => ptest
        )
    )
    
    npart_total = MPI.Allreduce(A.npart_local, MPI.SUM, comm)
    @assert npart_total == N * size
    
    for dx in 1:domain.ndim
        lower, upper = get_subdomain_bounds(domain, dx)
        @test maximum(lower .- A["P"][:, dx]) <= 0
        @test minimum(A["P"][:, dx] .- upper) <= 0
    end


end
