export rand_within_extents

function rand_within_extents(N::Int64, extent)
    
    ndim = length(extent)
    output = rand(Float64, (N, ndim))
    for dx in 1:ndim
        output[:, dx] .*= extent[dx]
    end

    return output
end
