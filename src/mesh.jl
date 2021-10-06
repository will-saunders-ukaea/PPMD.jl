export MinimalWidthCartesianMesh, mesh_hash, cellid_particle_dat_name, cellid_particle_dat


"""
A mesh constructed after domain decomposition consisting of cells that are at
least a certain width in each dimension.
"""
mutable struct MinimalWidthCartesianMesh
    domain
    min_width
    cell_dim
    cell_count
    hash
    function MinimalWidthCartesianMesh(domain, min_width)
        
        cell_dim = [0 for ix in 1:domain.ndim]
        for dx in 1:domain.ndim
            lower, upper = get_subdomain_bounds(domain, dx)
            local_extent = upper - lower
            ncells = Int64(floor(local_extent / min_width))
            @assert ncells > 0
            cell_dim[dx] = ncells
        end
        cell_count = reduce(*, cell_dim)
        
        new_mesh = new(domain, min_width, cell_dim, cell_count)
        
        hash = "$(typeof(new_mesh))_" * hash_primitive_type(min_width)
        new_mesh.hash = hash

        return new_mesh
    end
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
    if !(dat_name in keys(particle_group.particle_dats))
        add_particle_dat(particle_group, dat_name, ParticleDat(1, Int64))
    end
    dat = particle_group[dat_name]
    @assert dat.ncomp == 1
    @assert dat.dtype == Int64
    return dat
end
