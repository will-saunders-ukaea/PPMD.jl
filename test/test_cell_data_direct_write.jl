using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI


@testset "test_direct_access_write_1 $spec" for spec in (KACPU(), KACUDADevice(),)

    target_device = spec
    
    N = 25700

    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(3, position=true),
             "A" => ParticleDat(6, Int64),
        ),
        target_device
    )
    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
        )
    )

    mesh = MinimalWidthCartesianMesh(domain, 0.1)
    C = CellDat(mesh, (1,) , Int64, target_device)
    cdata = [ix for ix in 1:6]
    C[:] .= 0


    kernel_copy = Kernel(
        "direct_access_write_kernel",
        """
        for jx in 1:6
            ATOMIC_ADD(C, jx, jx)
        end
        """
    )
 

    loop = ParticleLoop(
        target_device,
        kernel_copy,
        (
            Dict(
                "A" => (A["A"], READ),
                "C" => (DirectAccess(C), WRITE),
            )
        )
    )

    execute(loop)

    @test norm(C[1:6] .- A.npart_local * cdata, Inf) < 1E-14

end
