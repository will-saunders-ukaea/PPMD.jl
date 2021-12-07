using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI

@testset "local_transfer_1 $spec" for spec in (KACPU(), )
#@testset "local_transfer_1 $spec" for spec in (KACPU(), KACUDADevice())

    target_device = spec

    N = 10

    extents = (4.0, )
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(1, position=true),
             "B" => ParticleDat(1),
             "A" => ParticleDat(1),
        ),
        target_device
    )
    
    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
             "A" => rand(Float64, (N, 1))
        )
    )

    @show MPI.Comm_rank(domain.comm)

    neigbour_ranks = get_neighbour_ranks(domain, 1)

    @show neigbour_ranks

    setup_local_transfer(A, neigbour_ranks)


    
end
