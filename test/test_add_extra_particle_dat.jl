using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI


@testset "test_add_extra_particle_dat_1 $spec" for spec in (KACPU(),KACUDADevice())

    target_device = spec
    
    N = 257

    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(3, position=true),
        ),
        target_device
    )

    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
        )
    )

    add_particle_dat(A, "C", ParticleDat(1, Int64))

    @test A["C"].npart_local == A["P"].npart_local
    @test size(A["P"][:, :])[1] == size(A["C"][:, :])[1]
    
    free(A)
end
