using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI

@testset "test_global_array_write_1 $spec" for spec in (KACPU(), KACUDADevice())

    target_device = spec
    
    # prime to ensure not multiple of workgroup size
    N = 257

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
             #"A" => rand(Float64, (N, 1))
             "A" => reshape([1 for ix in 1:N], (N, 1)),
        )
    )

    C = GlobalArray(2, target_device)


    kernel_copy = Kernel(
        "copy_kernel",
        """
        C[1] += A[ix, 1]
        C[2] += A[ix, 1] * 2.0
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

    execute(loop)

       
    host_reduce = MPI.Allreduce([sum(A["A"][:, 1]),], MPI.SUM, MPI.COMM_WORLD)

    @test abs(C[1] - host_reduce[1]) < 1E-13
    @test abs(C[2] - 2.0 * host_reduce[1]) < 1E-13

    execute(loop)

    @test abs(C[1] - 2.0 * host_reduce[1]) < 1E-12
    @test abs(C[2] - 4.0 * host_reduce[1]) < 1E-12

end
