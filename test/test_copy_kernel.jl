using PPMD
using Test
using CUDA
using LinearAlgebra


@testset "CPU_copy_kernel $spec" for spec in (KACPU(), KACUDADevice())

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
             "C" => ParticleDat(1),
        ),
        target_device
    )
    


    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
             "A" => rand(Float64, (N, 1)),
             "C" => rand(Float64, (N, 1))
        )
    )


    kernel_copy = Kernel(
        "copy_kernel",
        """
        B[ix, 1] = A[ix, 1];
        B[ix, 2] = A[ix, 1] * 2.0;
        B[ix, 3] = A[ix, 1] * 3.0;
        """
   )


    loop = ParticleLoop(
        target_device,
        kernel_copy,
        (
            Dict(
                "B" => (A["B"], WRITE),
                "A" => (A["A"], READ),
                "C" => (A["C"], INC_ZERO),
            )
        )
    )

    execute(loop)
    
    @test norm(A["C"][:, 1], Inf) < 1E-14
    @test norm(A["A"][:, 1]       - A["B"][:, 1], Inf) < 1E-14
    @test norm(A["A"][:, 1] * 2.0 - A["B"][:, 2], Inf) < 1E-14
    @test norm(A["A"][:, 1] * 3.0 - A["B"][:, 3], Inf) < 1E-14
    
    free(A)
end
