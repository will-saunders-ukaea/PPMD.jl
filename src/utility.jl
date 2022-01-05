export rand_within_extents, get_subdomain_bounds, get_neighbour_ranks, get_stencil_width

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


"""
Get the MPI ranks within a certain offset size on the cart comm of a
StructuredCartesianDomain.
"""
function get_neighbour_ranks(domain::StructuredCartesianDomain, stencil_width::Int64)::Array{Cint}
    
    comm = domain.comm
    coords = MPI.Cart_coords(comm)
    rank = MPI.Comm_rank(comm)
    dim = length(coords)
    
    # for each dimension create the range of possible ranks
    iterset = [
        convert(
            Array{Cint}, 
            (coords[dx] - stencil_width) : (coords[dx] + stencil_width) 
        ) for dx in 1:dim
    ]
    
    ranks = Set{Cint}()
    # find the remote ranks - assumes periodic domain
    tmp_array = zeros(Cint, dim)
    for coordx in Base.Iterators.product(iterset...)
        for dx in 1:dim
            tmp_array[dx] = coordx[dx]
        end
        remote_rank = MPI.Cart_rank(comm, tmp_array)
        if remote_rank != rank
            push!(ranks, remote_rank)
        end
    end
        
    n_ranks = length(ranks)
    ranks_array = Array{Cint}(undef, n_ranks)
    for rx in 1:n_ranks
        ranks_array[rx] = pop!(ranks)
    end

    return ranks_array
end


"""
Get the stencil width (the number of MPI ranks) required to cover a halo of a
given width.
"""
function get_stencil_width(domain::StructuredCartesianDomain, width)::Int64
    width = Float64(width)
    
    ndim = domain.ndim

    min_width = prevfloat(typemax(Float64))
    for dx in 1:ndim
        l, u = get_subdomain_bounds(domain, dx)
        w = u - l
        @assert w > 0.0
        min_width = min(min_width, w)
    end

    stencil_width = max(1, Int64(ceil(width / min_width)))

    return stencil_width

end





