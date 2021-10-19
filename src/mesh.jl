export MinimalWidthCartesianMesh, get_map_particle_to_cell_task


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
    lower_bounds
    upper_bounds

    function MinimalWidthCartesianMesh(domain, min_width)
        
        ndim = domain.ndim
        cell_dim = [0 for ix in 1:ndim]
    
        lower_bounds = zeros(Float64, ndim)
        upper_bounds = zeros(Float64, ndim)

        for dx in 1:ndim
            lower, upper = get_subdomain_bounds(domain, dx)
            local_extent = upper - lower
            ncells = Int64(floor(local_extent / min_width))
            @assert ncells > 0
            cell_dim[dx] = ncells

            lower_bounds[dx] = lower
            upper_bounds[dx] = upper
        end
        cell_count = reduce(*, cell_dim)
        
        new_mesh = new(domain, min_width, cell_dim, cell_count)
        
        # compute the hash for this mesh
        hash = "$(typeof(new_mesh))_" * hash_primitive_type(min_width)
        new_mesh.hash = hash

        # construct the data required to bin particles into cells
        new_mesh.lower_bounds = lower_bounds
        new_mesh.upper_bounds = upper_bounds

        return new_mesh
    end
end


"""
Create a task, e.g. ParticleLoop, that maps particle positions to cell indices.
"""
function get_map_particle_to_cell_task(mesh::MinimalWidthCartesianMesh, particle_group::ParticleGroup)
    
    
    ndim = particle_group.domain.ndim

    src = ""
    for dx in 1:ndim
        cell_width_dx = (mesh.upper_bounds[dx] - mesh.lower_bounds[dx]) / mesh.cell_dim[dx]
        i_cell_width_dx = 1.0 / cell_width_dx
        src_dx = """
        cell_$dx = FLOAT64_TO_INT64(Float64(trunc((POS[ix, $dx] - LOWER[$dx]) * $i_cell_width_dx)))
        if ((cell_$dx == $(mesh.cell_dim[dx])) & (abs(UPPER[$dx] - POS[ix, $dx]) < 1E-10))
            cell_$dx = $(mesh.cell_dim[dx] - 1)
        end
        """
        src *= src_dx
    end
    
    src  *= "\nlin_$ndim = cell_$ndim"
    for dx in ndim-1:-1:1
        src *= "\nlin_$dx = cell_$dx + $(mesh.cell_dim[dx]) * lin_$(dx+1)"
    end
    src *= """\n
    CELL[ix, 1] = lin_1 + 1
    """

    lower_ga = GlobalArray(ndim, Float64, particle_group.compute_target)
    lower_ga[:] = mesh.lower_bounds[:]
    upper_ga = GlobalArray(ndim, Float64, particle_group.compute_target)
    upper_ga[:] = mesh.upper_bounds[:]

    bin_kernel = Kernel(
        "_kernel_bin_MinimalWidthCartesianMesh",
        src
    )

    bin_loop = ParticleLoop(
        particle_group.compute_target,
        bin_kernel,
        Dict(
             "POS" => (particle_group[particle_group.position_dat], READ),
             "CELL" => (cellid_particle_dat(particle_group, mesh), WRITE),
             "LOWER" => (lower_ga, READ),
             "UPPER" => (upper_ga, READ),
        )
    )
    bin_loop.data["lower_ga"] = lower_ga
    bin_loop.data["upper_ga"] = upper_ga
    
    return bin_loop

end
