using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI
using Random

CUDA.allowscalar(true)

@testset "test_pair_dof_part_basic_read_1 $spec" for spec in (KACPU(), KACUDADevice(),)

    target_device = spec
    
    N = 1270

    extents = (4.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(2, position=true),
             "Q" => ParticleDat(1),
        ),
        target_device
    )

    mesh = MinimalWidthCartesianMesh(domain, 1.0)


    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
        )
    )
    
    # set the cells
    cellid = cellid_particle_dat(A, mesh)
    rng = MersenneTwister(91241)
    random_cells = rand(rng, 1:mesh.cell_count, A.npart_local)
    cellid[:, 1] = random_cells
    
    # create some dof data to read
    CD_B = CellDat(mesh, (4, 1), Float64, target_device)
    for cx in 1:mesh.cell_count
        for dx in 1:4
            CD_B[cx, dx, 1] = cx*4+dx
        end
    end

    # loop over dofs and sum the values onto the particle
    kernel_dof_read = Kernel(
        "dof_read_example",
        """
        Q[ix, 1] += B[1]
        """
    )
    loop = PairLoop(
        target_device,
        kernel_dof_read,
        Dict(
            "B" => (CD_B, READ),
        ),
        Dict(
            "Q" => (A["Q"], INC),
        )
    )

    execute(loop)

    for px in 1:A.npart_local
        cell = cellid[px, 1]
        correct = sum(CD_B[cell, :, :])
        to_test = A["Q"][px, 1]
        @test abs(correct - to_test) < 1E-12
    end

    free(A)
end
