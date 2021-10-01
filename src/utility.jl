export rand_within_extents, get_subdomain_bounds

using Random

"""
Generate N uniform random positions between zero and the extents provided.
Dimenionality is determined by the dimension of extent.
"""
function rand_within_extents(N::Int64, extent, rng=MersenneTwister())

    ndim = length(extent)
    output = rand(rng, Float64, (N, ndim))
    for dx in 1:ndim
        output[:, dx] .*= extent[dx]
    end

    return output
end


"""
Get the upper and lower boundary of the owned subdomain of a
StructuredCartesianDomain.
"""
function get_subdomain_bounds(domain::StructuredCartesianDomain, dim)
    
    dims, periods, coords = MPI.Cart_get(domain.comm)

    width = domain.extent[dim] / dims[dim]

    lower = coords[dim] * width
    upper = lower + width

    return lower, upper

end
