export MinimalWidthCartesianMesh


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




