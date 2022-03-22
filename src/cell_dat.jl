export CellDat, resize_cell_dat, getindex, setindex, show

using CUDA
CUDA.allowscalar(true)


"""
A CellDat is a data structure to store arrays worth of information on a per
cell basis. This is useful for storing maps from cells to particle indices and
per cell data such as dofs/coefficients. TODO better describe non-constant
arity.
"""
mutable struct CellDat
    mesh
    ncomp
    dtype
    compute_target
    data
    stride
    ncomp_first_max
    function CellDat(mesh, ncomp, dtype, compute_target)
        
        @assert typeof(ncomp) <: Tuple

        if (typeof(ncomp[1]) == CellDat)
            @assert ncomp[1].mesh == mesh
            _ncomp = 0
            _ncomp_first_max = 0
        else
            _ncomp = ncomp
            _ncomp_first_max = ncomp[1]
        end

        required_length = (_ncomp..., mesh.cell_count,)
        stride = reduce(*, _ncomp)
        data = device_zeros(compute_target, dtype, required_length)

        return new(mesh, ncomp, dtype, compute_target, data, stride, _ncomp_first_max)
    end
end


"""
If a Determine the size of the iteration set required to iterated over all of
the DOFs stored in a CellDat.
"""
function get_iteration_set_size_mesh_dofs(cell_dat)
    # Either there are a constant number of DOFs per cell or there is a
    # variable number determined by an additional CellDat. In the second case
    # the stride (currently) is the maximum number of DOFs in any cell.
    return cell_dat.ncomp_first_max
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
        cell_dat.ncomp_first_max = max_var_ncomp
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
    
    #@assert typeof(key[1]) <: Int "Cannot slice across cells in CellDat indexing."

    key = (key[2:length(key)]... ,key[1])

    base_array = base_array[key...]
    if typeof(base_array) <: SubArray
        return convert(Array{dat.dtype}, base_array)
    else
        return base_array
    end

end


"""
Pretty(ish) printing of CellDats
"""
function Base.show(io::IO, cell_dat::CellDat)
    print(io, "\nncell_local: ", cell_dat.mesh.cell_count, "\n")
    print(io, "dtype:       ", cell_dat.dtype, "\n")
    print(io, "ncomp:       ", cell_dat.ncomp, "\n")
    if isdefined(cell_dat, :compute_target)
        print(io, "target:      ", cell_dat.compute_target, "\n")
    end
    if isdefined(cell_dat, :data)
        ncomp_slice = [(:) for nx in cell_dat.ncomp]
        for ix in 1:cell_dat.mesh.cell_count
            print(io, cell_dat[ix, ncomp_slice...], "\n")
        end
    end
end


"""
Zero the values in a CellDat
"""
function zero_dat(cell_dat::CellDat)
    fill!(cell_dat.data, 0.0)
end
