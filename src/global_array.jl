export GlobalArray, getindex

using CUDA
CUDA.allowscalar(true)


"""
GobalArray, an Array type that can be read or incremented in a Kernel.
"""
mutable struct GlobalArray
    ncomp::Int64
    dtype::DataType
    compute_target::Any
    data::Any
    function GlobalArray(ncomp::Int64, dtype::DataType, compute_target)
        return new(ncomp, dtype, compute_target, compute_target.ArrayType{dtype}(undef, ncomp))
    end
    function GlobalArray(ncomp::Int64, compute_target)
        return new(ncomp, Float64, compute_target, compute_target.ArrayType{Float64}(undef, ncomp))
    end
end


"""
Allow access to GlobalArray data using subscripts.
"""
function Base.getindex(dat::GlobalArray, key...)

    base_array = get_data_on_host(dat, dat.compute_target, (1:dat.ncomp,))
    base_array = base_array[key...]
    if typeof(base_array) <: SubArray
        return convert(Array{dat.dtype}, base_array)
    else
        return base_array
    end

end
