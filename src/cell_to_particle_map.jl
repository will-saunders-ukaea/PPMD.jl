export cellid_particle_dat_name, cellid_particle_dat, layer_particle_dat_name, layer_particle_dat, CellToParticleMap, assemble_cell_to_particle_map


using CUDA


"""
Get the ParticleDat that corresponds to a given name in a ParticleGroup.
Creates the ParticleDat if it does not exist. Checks the number of components
and data type.
"""
function get_add_dat(particle_group, name, ncomp, dtype)
    if !(name in keys(particle_group.particle_dats))
        add_particle_dat(particle_group, name, ParticleDat(ncomp, dtype))
    end
    dat = particle_group[name]
    @assert dat.ncomp == ncomp
    @assert dat.dtype == dtype
    return dat
end


"""
Get the name of the ParticleDat in a ParticleGroup that contains the cell
indices for a given mesh.
"""
function cellid_particle_dat_name(mesh)
    return "_CellId_" * mesh.hash
end


"""
Get the ParticleDat in a ParticleGroup which stores the cell ids for a
particular mesh. Creates the ParticleDat if it does not exist.
"""
function cellid_particle_dat(particle_group, mesh)
    dat_name = cellid_particle_dat_name(mesh)
    return get_add_dat(particle_group, dat_name, 1, Int64)
end


"""
Get the name of the ParticleDat in a ParticleGroup that contains the cell
layer for a given mesh.
"""
function layer_particle_dat_name(mesh)
    return "_Layer_" * mesh.hash
end


"""
Get the ParticleDat in a ParticleGroup which stores the layer for a
particular mesh. Creates the ParticleDat if it does not exist.
"""
function layer_particle_dat(particle_group, mesh)
    dat_name = layer_particle_dat_name(mesh)
    return get_add_dat(particle_group, dat_name, 1, Int64)
end




"""
Assume there exists a ParticleDat containing the owning cell of each particle.
Compute the layer each particle is on and the total number of particles in each
cell then build the cell to particle map.
"""
function assemble_cell_to_particle_map(map)

    execute(map.loop_layer)
    resize_cell_dat(map.cell_children)
    CUDA.@allowscalar map.layer_stride.dat.data[1] = map.cell_children.stride
    execute(map.loop_assemble)

end


"""
Contains objects required to map from cells to particles within that cell.
"""
mutable struct CellToParticleMap
    mesh
    particle_group
    cell_npart
    cell_children
    loop_layer
    layer_stride
    loop_assemble

    function CellToParticleMap(mesh, particle_group)
        compute_target = particle_group.compute_target
        new_map = new(mesh, particle_group)
        new_map.cell_npart = CellDat(mesh, (1,), Int64, compute_target)
        new_map.cell_children = CellDat(mesh, (new_map.cell_npart,), Int64, compute_target)

        # creates the cellid ParticleDat
        cellid = cellid_particle_dat(particle_group, mesh)
        # creates the layer ParticleDat
        layer = layer_particle_dat(particle_group, mesh)

        kernel_layer = Kernel(
            "bin_particles_into_cells_kernel",
            """
            # The +1 is for Julia's base 1 indexing.
            LAYER[ix, 1] = ATOMIC_ADD(CELL_COUNTS, CELL[ix, 1], 1) + 1
            """
        )
        new_map.loop_layer = ParticleLoop(
            compute_target,
            kernel_layer,
            (
                Dict(
                    "CELL" => (cellid, READ),
                    "LAYER" => (layer, WRITE),
                    "CELL_COUNTS" => (DirectAccess(new_map.cell_npart), WRITE),
                )
            )
        )

        new_map.layer_stride = DirectArray([new_map.cell_children.stride,], compute_target)

        kernel_assemble = Kernel(
            "populate_cell_to_particle_map",
            """
            CELL_MAP[(CELL[ix, 1] - 1) * LAYER_STRIDE[1] + LAYER[ix, 1]] = _LOCAL_RANK_INDEX
            """
        )
        new_map.loop_assemble = ParticleLoop(
            compute_target,
            kernel_assemble,
            (
                Dict(
                    "LAYER" => (layer, READ),
                    "CELL" => (cellid, READ),
                    "LAYER_STRIDE" => (new_map.layer_stride, READ),
                    "CELL_MAP" => (DirectAccess(new_map.cell_children), WRITE),
                )
            )
        )
        
        return new_map
    end

end


"""
Get a key for a cell to particle map using a given mesh.
"""
function cell_to_particle_map_hash(mesh)
    return "cell_to_particle_map_" * mesh.hash
end


"""
Get the map from cells to particles for a given Mesh and ParticleGroup. Creates
a new map if one does not exist already.
"""
function get_cell_to_particle_map(mesh, particle_group)
    
    hash = cell_to_particle_map_hash(mesh)
    @show hash
    if !haskey(particle_group.maps, hash)
        map = CellToParticleMap(mesh, particle_group)
        particle_group.maps[hash] = map
    else
        map = particle_group.maps[hash]
    end
    
    return map
end


"""
Assemble a cell to particle map if required.
"""
function assemble_map_if_required(map)
    
    # TODO use version ids to determine if reassembly is required.
    assemble_cell_to_particle_map(map)

end








