export MinimalWidthCartesianMesh




mutable struct MinimalWidthCartesianMesh
    domain
    min_width
    cell_dim
    cell_count
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

        return new(domain, min_width, cell_dim, cell_count)
    end

end




