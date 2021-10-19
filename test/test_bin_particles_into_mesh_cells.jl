using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI
using Random

CUDA.allowscalar(true)

@testset "test_cell_binning_1 $spec" for spec in (KACPU(),KACUDADevice(),)

    target_device = spec
    
    N = 1000

    extents = (2.0, 3.0, 5.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    ndim = domain.ndim

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(ndim, position=true),
             "Q" => ParticleDat(1),
        ),
        target_device
    )

    mesh = MinimalWidthCartesianMesh(domain, 0.123)


    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
             "Q" => ones(Float64, (N, 1))
        )
    )

    cell_bin_task = get_map_particle_to_cell_task(mesh, A)

    execute(cell_bin_task)
    cell_indices = cellid_particle_dat(A, mesh)
    
    domain_local_extents = mesh.upper_bounds[:] - mesh.lower_bounds[:]
    inverse_cell_widths = [1.0/(domain_local_extents[dx] / mesh.cell_dim[dx]) for dx in 1:ndim]
    
    P = A["P"]

    for px in 1:A.npart_local
        
        c1 = Int64(trunc((P[px, 1] - mesh.lower_bounds[1]) * inverse_cell_widths[1]))
        c2 = Int64(trunc((P[px, 2] - mesh.lower_bounds[2]) * inverse_cell_widths[2]))
        c3 = Int64(trunc((P[px, 3] - mesh.lower_bounds[3]) * inverse_cell_widths[3]))

        cell = 1 + c1 + mesh.cell_dim[1] * (c2 + mesh.cell_dim[2] * c3)
        
        @test cell == cell_indices[px, 1]
    end

end
