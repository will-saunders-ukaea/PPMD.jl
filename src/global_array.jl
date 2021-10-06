export GlobalArray, getindex, setindex, increment

using CUDA
using MPI
CUDA.allowscalar(true)


"""
GobalArray, an Array type that can be read or incremented in a Kernel.
"""
mutable struct GlobalArray
    ncomp::Int64
    dtype::DataType
    compute_target::Any
    data::Any
    comm::MPI.Comm
    function GlobalArray(ncomp::Int64, dtype::DataType, compute_target; comm=MPI.COMM_WORLD)
        return new(ncomp, dtype, compute_target, device_zeros(compute_target, dtype, ncomp), comm)
    end
    function GlobalArray(ncomp::Int64, compute_target; comm=MPI.COMM_WORLD)
        return new(ncomp, Float64, compute_target, device_zeros(compute_target, Float64, ncomp), comm)
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

"""
Allow writing to a GlobalArray
"""
function Base.setindex!(dat::GlobalArray, value, key...)
    CUDA.@allowscalar dat.data[key...] = value
end


"""
Increment the values in the array by another array.
"""
function increment(dat::GlobalArray, value::Array)
    dat.data[:] += dat.compute_target.ArrayType(value)
end


