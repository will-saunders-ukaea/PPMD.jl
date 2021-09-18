using PPMD
using Test
using CUDA
using LinearAlgebra


@testset "test_global_array_write_1 $spec" for spec in (KACPU(), KACUDADevice())

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
    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
             "A" => rand(Float64, (N, 1))
        )
    )

    C = GlobalArray(1, target_device)


    kernel_copy = Kernel(
        "copy_kernel",
        """
        C[1] += A[ix, 1]
        """
    )
 

    loop = ParticleLoop(
        target_device,
        kernel_copy,
        (
            Dict(
                "A" => (A["A"], READ),
                "C" => (C, INC),
            )
        )
    )

    #execute(loop)

end
