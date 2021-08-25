export FullyPeroidicBoundary, StructuredCartesianDomain

abstract type BoundaryCondition end
abstract type Domain end


struct FullyPeroidicBoundary <: BoundaryCondition

end



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
