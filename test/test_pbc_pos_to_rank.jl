using MPI
using PPMD
using Test
using LinearAlgebra



#@testset "PBC position to rank $spec" for spec in (KACUDADevice(),)
@testset "PBC position to rank $spec" for spec in (KACPU(), KACUDADevice())

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

    kernel, dat_mapping = get_position_to_rank_kernel(A.domain, A["B"], A["_owning_rank"])


    loop = ParticleLoop(
        A.compute_target,
        kernel,
        dat_mapping
    )

    execute(loop)


    proposed_ranks = A["_owning_rank"][:, :]

    dims, periods, coords = MPI.Cart_get(domain.comm)

    width_1 = extents[1] / dims[1]
    width_2 = extents[2] / dims[2]
    width_3 = extents[3] / dims[3]

    for px in 1:A.npart_local
        
        pos = A["B"][px, 1:3]

        c1 = trunc(Int64, pos[1] / width_1)
        c2 = trunc(Int64, pos[2] / width_2)
        c3 = trunc(Int64, pos[3] / width_3)
        
        rank = c3 + dims[3] * (c2 + dims[2] * c1)
        p_rank = Int64(proposed_ranks[px, 1])

        @test rank == p_rank

    end

end
