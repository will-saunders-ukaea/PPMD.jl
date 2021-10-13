export PairLoop

using KernelAbstractions
using FunctionWrappers: FunctionWrapper
using DataStructures


function arg_types(args)
    type_args = Set((typeof(args[keyx][1]) for keyx in keys(args)))
    return type_args
end


"""
Construct the parameters for the kernel function
"""
function get_wrapper_param(kernel_sym, dat::CellDat, access_mode, target)
    return ("_global_$kernel_sym",)
end


"""
Generate code for the data structure and access type prior to the kernel launch 
"""
function get_pre_kernel_launch(kernel_sym, dat::CellDat, access_mode, target)
    remaining_dims = join((":" for dx in 1:length(dat.ncomp)-1), ",")
    return "$kernel_sym = view(_global_$kernel_sym, _global_dof, $remaining_dims )"
end


"""
Determine if a Dict of args contains a write access descriptor.
"""
function has_write_access(args)
    return reduce(|, (ax.second[2].write for ax in args))
end


"""
Determine if a Dict of args is ParticleDat based or CellDat based.
"""
function args_dat_type(args)
    for ax in args
        t = typeof(ax.second[1])
        if t in (ParticleDat, CellDat)
            return t
        end
    end
end


"""
Find a CellDat in the args
"""
function find_cell_dat(args)
    for ax in args
        dat = ax.second[1]
        if typeof(dat) == CellDat
            return dat
        end
    end
end


"""
Find the ParticleDat from the args
"""
function find_particle_dat(args)
    for ax in args
        dat = ax.second[1]
        if typeof(dat) == ParticleDat
            return dat
        end
    end
end


"""
A PairLoop between CellDats and ParticleDats on a per cell basis.
"""
function DOFParticlePairLoop(
    target,
    kernel,
    args_a,
    args_b
)
    @assert arg_types(args_a) == Set((CellDat,))

    # ensure args have a consistent ordering
    args_a = make_args_sorted(args_a)
    args_b = make_args_sorted(args_b)
    args = merge(args_a, args_b)

    first_cell_dat = find_cell_dat(args)
    mesh = first_cell_dat.mesh
    first_particle_dat = find_particle_dat(args)
    particle_group = first_particle_dat.particle_group

    # create the map from cells to particles
    cell_to_particle_map = CellToParticleMap(mesh, particle_group)
    particle_group.maps[mesh.hash] = cell_to_particle_map
    
    # loop over the args which have write access as the first loop
    if has_write_access(args_a)
        args_first = args_a
        args_second = args_b
    else
        args_first = args_b
        args_second = args_a
    end
    args_main_types = (args_dat_type(args_first), args_dat_type(args_second))

    if args_main_types == (CellDat, ParticleDat)
        # first loop is over DOFs second over particles
        init_loop_1 = """
        # assume _ix is at least ncell * max_n_dof

        # compute the cell containing the DOF
        _cellx = div(_ix, _C2DOF_STRIDE)
        # compute the DOF this interation touches
        _dofx = mod(_ix, _C2DOF_STRIDE)
        
        # mask off the dofs that don't exist
        if (_dofx > _C2DOFCOUNT_MAP[_cellx])
            return
        end

        # currently this storage is a matrix 
        _global_dofx = _ix
        """
        finalise_loop_1 = ""
        init_loop_2 = """
        # loop over particles

        _npart_in_cell = _CELL_PARTICLE_COUNTS[_cellx]
        for _layerx in 1:_npart_in_cell
            ix = _C2P_MAP[_cellx * _C2P_MAP_STRIDE + _layerx]
        """
        finalise_loop_2 = "end"

        # function to get the size of the iteration set
        first_iteration_size = () -> return mesh.cell_count * get_iteration_set_size_mesh_dofs(first_cell_dat)

        # function to run to build cell to particle maps
        init_particle_map = () -> return assemble_cell_to_particle_map(cell_to_particle_map)
    else
        # first loop is over particles second over dofs
        init_loop_1 = """
        # assume _ix is npart_local
        ix = _ix
        """
        finalise_loop_1 = ""
        init_loop_2 = """
        # loop over cell dofs
        _cell = _P2C_MAP[ix, 1]
        _start_dof = ((_cellx-1) * _C2DOF_STRIDE)
        _end_dof = _start_dof + _C2DOFCOUNT_MAP[_cellx]
        for _global_dofx in _start_dof:_end_dof
        """
        finalise_loop_2 = "end"

        # function to get the size of the iteration set
        first_iteration_size = () -> return particle_group.npart_local

        # Do not need the map from cells to particles in this case
        # only particles to cells
        init_particle_map = () -> return true
    end
    
    # Assemble the kernel function parameters
    kernel_params = flatten_join(
        [get_wrapper_param(px.first, px.second[1], px.second[2], target) for px in args],
        ','
    )
    pre_kernel_launch = [join(
        [get_pre_kernel_launch(px.first, px.second[1], px.second[2], target) for px in arg_set], 
        "\n"
    ) for arg_set in (args_first, args_second)]


    post_kernel_launch = ""
    # is a syncronize required before the kernel call? e.g. for local_mem init
    # TODO implement support for GlobalArray reductions on this loop type
    if Bool(false)
        pre_kernel_sync = "@synchronize"
    else
        pre_kernel_sync = ""
    end


    # Assemble the kernel function
    kernel_func = """
    @kernel function kernel_pair_wapper(
        _NPART_LOCAL,
        _NCELL_LOCAL,
        _C2DOFCOUNT_MAP,
        _C2DOF_STRIDE,
        _CELL_PARTICLE_COUNTS,
        _C2P_MAP_STRIDE,
        _C2P_MAP,
        _P2C_MAP,
        $kernel_params
    )

        ATOMIC_ADD = $(get_atomic_add(target))

        _local_ix = @index(Local)
        _group_ix = @index(Group)
        _ix = @index(Global)
        
        $init_loop_1 # init loop 1 - loop over dofs or particles
        
            $(pre_kernel_launch[1])
            $pre_kernel_sync

            @inbounds begin
                $init_loop_2 # init loop 2

                    $(pre_kernel_launch[2])

                    $(kernel.source)

                $finalise_loop_2 #end loop 2
            end
        $finalise_loop_1 # end loop 1

        $post_kernel_launch

    end
    """

    println(kernel_func)

    l = Task(
        kernel.name * "_" * string(target) * "_ParticleLoop",
        () -> return
    )

    # Pass the kernel function to KernelAbstractions
    @eval new_kernel() = $(Meta.parse(kernel_func))
    
    ka_methods = Base.invokelatest(new_kernel)
    loop_func = Base.invokelatest(ka_methods, target.device, target.workgroup_size)


end


function PairLoop(
    target,
    kernel,
    args_a,
    args_b
)
    return DOFParticlePairLoop(target, kernel, args_a, args_b)
end


