using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI
using Random

CUDA.allowscalar(true)

@testset "test_pair_dof_part_basic_write_1 $spec" for spec in (KACUDADevice(),KACPU(),)

    target_device = spec
    
    N = 10

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
             "Q" => ones(Float64, (N, 1))
        )
    )
    
    # set the cells
    cellid = cellid_particle_dat(A, mesh)
    rng = MersenneTwister(91241)
    random_cells = rand(rng, 1:mesh.cell_count, A.npart_local)
    cellid[:, 1] = random_cells
    
    # create some dof data to read
    ndof = 2
    CD_A = CellDat(mesh, (ndof, 2), Float64, target_device)
    CD_B = CellDat(mesh, (ndof, 1), Float64, target_device)

    CD_A[:, 1, 1] = 3.0 * ones(mesh.cell_count)
    CD_A[:, 1, 2] = 4.0 * ones(mesh.cell_count)
    CD_A[:, 2, 1] = 13.0 * ones(mesh.cell_count)
    CD_A[:, 2, 2] = 14.0 * ones(mesh.cell_count)    

    # loop over particles and sum the values onto the dof
    kernel_dof_read = Kernel(
        "dof_read_example",
        """
        B[1] += (A[1] + A[2]) * Q[ix, 1]
        """
    )
    loop = PairLoop(
        target_device,
        kernel_dof_read,
        Dict(
            "A" => (CD_A, READ),
            "B" => (CD_B, INC),
        ),
        Dict(
            "Q" => (A["Q"], READ),
        )
    )

    execute(loop)

    correct_array = zeros(Float64, (mesh.cell_count, ndof, 1))
    for px in 1:A.npart_local
        cell = cellid[px, 1]
        correct_array[cell, 1] += 7.0 
        correct_array[cell, 2] += 27.0 
    end

    for cellx in 1:mesh.cell_count
        @test norm(CD_B[cellx, :, 1] - correct_array[cellx, :], Inf) < 1E-12
    end
    
    free(A)
end
