using PPMD
using Test
using CUDA
using LinearAlgebra
using MPI
using Random

CUDA.allowscalar(true)

@testset "test_cell_to_particle_map_construction_1 $spec" for spec in (KACPU(),KACUDADevice())

    target_device = spec
    
    # prime to ensure not multiple of workgroup size
    N = 1024

    extents = (4.0, 4.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(3, position=true),
        ),
        target_device
    )

    mesh = MinimalWidthCartesianMesh(domain, 1.0)

    cell_npart = CellDat(mesh, 1, Int64, target_device)
    cell_children = CellDat(mesh, cell_npart, Int64, target_device)

    add_particles(
        A,
        Dict(
             "P" => rand_within_extents(N, domain.extent),
        )
    )
    
    # creates the cellid ParticleDat
    cellid = cellid_particle_dat(A, mesh)

    rng = MersenneTwister(91241)
    random_cells = rand(rng, 1:mesh.cell_count, A.npart_local)
    cellid[:, 1] = random_cells

    map = CellToParticleMap(mesh, A)

    assemble_cell_to_particle_map(map)

    stride = map.layer_stride.dat.data[1]
    
    npart_seen = 0
    for cellx in 1:mesh.cell_count
        
        cstart = (cellx - 1) * stride + 1
        cend = cstart + map.cell_npart[cellx] - 1

        for px in map.cell_children[cstart:cend]
            @test cellid[px, 1] == cellx
            npart_seen += 1
        end


    end

    @test npart_seen == A.npart_local

end
