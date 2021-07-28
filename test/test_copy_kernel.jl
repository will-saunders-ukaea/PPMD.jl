using jPPMD
using Test
using CUDA
using LinearAlgebra

@testset "CPU_copy_kernel $spec" for spec in ((KADevice(KACUDADevice(), 16), CuArray), (KADevice(KACPU(), 16), Array))

    target_device = spec[1]
    ArrayType = spec[2]

    N = 10
    A = ArrayType{Float64}(undef, N, 1)
    B = ArrayType{Float64}(undef, N, 3)

    A[:] = rand(Float64, (N, 1))
    B[:] .= 0.0

    kernel_vv_2 = """
        B[ix, 1] = A[ix, 1];
        B[ix, 2] = A[ix, 1] * 2.0;
        B[ix, 3] = A[ix, 1] * 3.0;
    """

    loop = ParticleLoop(
        target_device,
        kernel_vv_2,
        (
            Dict(
                "A" => (A, READ),
                "B" => (B, WRITE),
            )
        )
    )

    execute(loop)
    
    @test norm(A[:]       - B[:, 1], Inf) < 1E-14
    @test norm(A[:] * 2.0 - B[:, 2], Inf) < 1E-14
    @test norm(A[:] * 3.0 - B[:, 3], Inf) < 1E-14

end
