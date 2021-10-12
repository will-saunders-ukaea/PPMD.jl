export CellDat, resize_cell_dat, getindex, setindex

using CUDA
CUDA.allowscalar(true)

mutable struct CellDat
    mesh
    ncomp
    dtype
    compute_target
    data
    stride
    function CellDat(mesh, ncomp, dtype, compute_target)
        
        @assert typeof(ncomp) <: Tuple

        if (typeof(ncomp[1]) == CellDat)
            @assert ncomp[1].mesh == mesh
            _ncomp = 0
        else
            _ncomp = ncomp
        end

        required_length = (_ncomp..., mesh.cell_count,)
        stride = reduce(*, _ncomp)
        data = device_zeros(compute_target, dtype, required_length)

        return new(mesh, ncomp, dtype, compute_target, data, stride)
    end
end


"""
Resize a CellDat to allocate sufficient storage for the current value of ncomp.
This is used when the number of components (ncomp) is itself a CellDat and
hence the number of required components is variable.
"""
function resize_cell_dat(cell_dat)
    
    ncomp = cell_dat.ncomp
    current_total_length = length(cell_dat.data)
    cell_count = cell_dat.mesh.cell_count
    # Case where ncomp is explicitly set to a fixed ncomp

    if !(typeof(ncomp[1]) == CellDat)
        @assert current_total_length == cell_data.stride * cell_count
    else
        max_var_ncomp = maximum(ncomp[1].data)
        cell_dat.data = device_zeros(
            cell_dat.compute_target, 
            cell_dat.dtype, 
            (max_var_ncomp, ncomp[2:length(ncomp)]..., cell_count)
        )
        if length(ncomp) > 1
            comp_factor = reduce(*, ncomp[2:length(ncomp)])
        else
            comp_factor = 1
        end

        cell_dat.stride = max_var_ncomp * comp_factor
    end

end


"""
Allow writing to a CellDat
"""
function Base.setindex!(dat::CellDat, value, key...)
    key = (key[2:length(key)]... ,key[1])
    CUDA.@allowscalar dat.data[key...] = value
end


"""
Allow access to CellDat data using subscripts.
"""
function Base.getindex(dat::CellDat, key...)

    base_array = get_data_on_host(dat, dat.compute_target, Tuple(UnitRange(1, rx) for rx in size(dat.data)))
    
    key = (key[2:length(key)]... ,key[1])

    base_array = base_array[key...]
    if typeof(base_array) <: SubArray
        return convert(Array{dat.dtype}, base_array)
    else
        return base_array
    end

end


