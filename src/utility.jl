export rand_within_extents


"""
Generate N uniform random positions between zero and the extents provided.
Dimenionality is determined by the dimension of extent.
"""
function rand_within_extents(N::Int64, extent)
    
    ndim = length(extent)
    output = rand(Float64, (N, ndim))
    for dx in 1:ndim
        output[:, dx] .*= extent[dx]
    end

    return output
end
