export CellDat, resize_cell_dat


mutable struct CellDat
    mesh
    ncomp
    dtype
    compute_target
    data
    stride
    function CellDat(mesh, ncomp, dtype, compute_target)
        
        if (typeof(ncomp) == CellDat)
            @assert ncomp.mesh == mesh
            _ncomp = 0
        else
            _ncomp = ncomp
        end

        required_length = _ncomp * mesh.cell_count
        data = device_zeros(compute_target, dtype, required_length)

        return new(mesh, ncomp, dtype, compute_target, data, _ncomp)
    end
end


function resize_cell_dat(cell_dat)
    
    ncomp = cell_dat.ncomp
    current_total_length = length(cell_dat.data)
    cell_count = cell_dat.mesh.cell_count
    # Case where ncomp is explicitly set to a fixed ncomp

    if !(typeof(ncomp) == CellDat)
        @assert current_total_length == ncomp * cell_count
    else
        max_ncomp = maximum(ncomp.data)
        required_length = max_ncomp * cell_count
        resize!(cell_dat.data, required_length)
        cell_dat.stride = max_ncomp
    end

end
