export FullyPeroidicBoundary, StructuredCartesianDomain, get_boundary_condition_loop, get_position_to_rank_loop, get_position_to_rank_kernel, global_move

using MPI

"Abstract type for boundary conditions"
abstract type BoundaryCondition end

"Abstract type for domains"
abstract type Domain end


"Fully periodic boundary conditions."
struct FullyPeroidicBoundary <: BoundaryCondition

end


"""
Create a cartcomm where the larger comm dimensions are commensurate with the
larger domain dimensions.
"""
function create_cartcomm(extent, comm)
    size = MPI.Comm_size(comm)
    ndim = length(extent)
    dims = zeros(Cint, (ndim,))

    MPI.Dims_create!(size, dims)
    
    order_dims = sortperm(dims)
    order_extent = sortperm(extent)
    
    new_dims = similar(dims)
    for dx in 1:ndim
        new_dims[order_extent[dx]] = dims[order_dims[dx]]
    end

    new_comm = MPI.Cart_create(comm, new_dims, ones(Cint, (ndim, )), true)

    return new_comm
end


"""
A cuboid domain that is decomposed into uniformly sized and shaped cuboids.
Instances are constructed with a boundary condition, e.g.
FullyPeroidicBoundary, and an extent.
"""
mutable struct StructuredCartesianDomain <: Domain
    boundary_condition
    extent::Array{Float64}
    ndim::Int64
    comm
    function StructuredCartesianDomain(bc, extent; comm=MPI.COMM_WORLD)
        @assert (typeof(bc) <: BoundaryCondition)
        extent = convert(Array{Float64}, collect(extent))
        ndim = length(extent)
        if (comm == MPI.COMM_WORLD)
            comm = create_cartcomm(extent, comm)
        end
        return new(bc, extent, ndim, comm)
    end
end


"""
Returns a ParticleLoop that applies periodic boundary conditions to the
particle positions.
"""
function get_boundary_condition_loop(boundary_condition::T, particle_group) where (T <: FullyPeroidicBoundary)

    position_dat = particle_group[particle_group.position_dat]
    domain = particle_group.domain
    extent = domain.extent
    ndim = domain.ndim
    
    kernel_pbc = ""
    for dx in 1:ndim
        kernel_pbc *= "\nP[ix, $dx] = (P[ix, $dx] + ceil(abs(P[ix, $dx] / $(extent[dx]))) * $(extent[dx])) % $(extent[dx])"
    end

    kernel_pbc = Kernel(
        "PBC_apply",
        kernel_pbc
    )

    loop = ParticleLoop(
        particle_group.compute_target,
        kernel_pbc,
        Dict(
            "P" => (position_dat, WRITE)
        )
    )

    return loop
end


"""
Return a kernel for a ParticleLoop that maps particle position to a MPI rank.
"""
function get_position_to_rank_kernel(domain::T, position_dat, rank_dat) where (T <: StructuredCartesianDomain)

    dims, periods, coords = MPI.Cart_get(domain.comm)
    
    src = ""
    for dx in 1:domain.ndim
        
        cell_width_dx = domain.extent[dx] / dims[dx]
        i_cell_width_dx = 1.0 / cell_width_dx
    

        src_dx = """
        cell_$dx = trunc(P[ix, $dx] * $i_cell_width_dx)
        """

        src *= src_dx

    end
    
    src  *= "\nlin_1 = cell_1"
    for dx in 2:domain.ndim
        src *= "\nlin_$dx = cell_$dx + $(dims[dx]) * lin_$(dx-1)"
    end
    src *= """\n
    rank = lin_$(domain.ndim)
    RANK[ix, 1] = rank
    """

    kernel = Kernel(
        "StructuredCartesianPositionToRank",
        src
    )

    dat_mapping = Dict(
        "P" => (position_dat, READ),
        "RANK" => (rank_dat, WRITE),
    )

    return kernel, dat_mapping

end


"""
Return the ParticleLoop Task that when executed maps positions to MPI ranks.
"""
function get_position_to_rank_loop(particle_group)

    kernel, dat_mapping = get_position_to_rank_kernel(
        particle_group.domain,
        particle_group[particle_group.position_dat],
        particle_group["_owning_rank"]
    )

    loop = ParticleLoop(
        particle_group.compute_target,
        kernel,
        dat_mapping
    )

    return loop
end

















