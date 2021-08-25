export FullyPeroidicBoundary, StructuredCartesianDomain

"Abstract type for boundary conditions"
abstract type BoundaryCondition end

"Abstract type for domains"
abstract type Domain end


"Fully periodic boundary conditions."
struct FullyPeroidicBoundary <: BoundaryCondition

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
    function StructuredCartesianDomain(bc, extent)
        @assert (typeof(bc) <: BoundaryCondition)
        extent = convert(Array{Float64}, collect(extent))
        ndim = length(extent)
        return new(bc, extent, ndim)
    end
end
