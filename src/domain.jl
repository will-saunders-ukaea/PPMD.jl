export FullyPeroidicBoundary, StructuredCartesianDomain

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
