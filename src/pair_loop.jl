export PairLoop

using KernelAbstractions
using FunctionWrappers: FunctionWrapper
using DataStructures
using Profile

function arg_types(args)
    type_args = Set((typeof(args[keyx][1]) for keyx in keys(args)))
    return type_args
end


"""
Construct the parameters for the kernel function
"""
function get_wrapper_param(kernel_sym, dat::CellDat, access_mode, target)
    return (const_modifier("_global_$kernel_sym", access_mode),)
end


"""
Generate code for the data structure and access type prior to the kernel launch 
"""
function get_pre_kernel_launch(kernel_sym, dat::CellDat, access_mode, target)
    remaining_dims = join((":" for dx in 1:length(dat.ncomp)-1), ",")
    return "$kernel_sym = view(_global_$kernel_sym, _dofx, $remaining_dims, _cellx )"
end


"""
Get the calling arguments for the data structure and access mode. This may 
also handle any communication required before the loop launches. e.g.
halo exchanges.
"""
function get_loop_args(N, kernel_sym, dat::CellDat, access_mode, target)
    return (dat.data,)
end


"""
After a loop has completed perform actions for each data structure and access
type, e.g. reduction operations.
"""
function post_loop(N, kernel_sym, dat::CellDat, access_mode, target, arg)
    return
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
Get the map from cells to DOFs counts. Finds which args contain the DOFs and
validates the map is the same for those CellDats.
"""
function find_cell_to_dof_map(args_a, args_b)
    for args in (args_a, args_b)
        if args_dat_type(args) == CellDat
            map = nothing
            max_dof_dat = nothing
            for argx in args
                dat = argx.second[1]
                if typeof(dat) == CellDat
                    if map == nothing
                        map = dat.ncomp[1]
                        max_dof_dat = dat
                    else
                        @assert dat.ncomp[1] == map
                    end
                end
            end
            
            get_max_dof_count = () -> return max_dof_dat.ncomp_first_max

            return map, get_max_dof_count
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
    cell2dofcount_map, get_max_dof_count = find_cell_to_dof_map(args_a, args_b)

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


    if typeof(cell2dofcount_map) == CellDat
        dof_count_lookup = "_C2DOFCOUNT_MAP[_cellx]"
    else
        dof_count_lookup = "_C2DOF_STRIDE"
    end


    if args_main_types == (CellDat, ParticleDat)
        # first loop is over DOFs second over particles
        init_loop_1 = """
        # assume _ix is at least ncell * max_n_dof

        # compute the cell containing the DOF
        _cellx = div(_ix-1, _C2DOF_STRIDE) + 1
        # compute the DOF this iteration touches
        _dofx = mod(_ix-1, _C2DOF_STRIDE) + 1
        """
        finalise_loop_1 = ""
        init_loop_2 = """
        # loop over particles

        _npart_in_cell = _CELL_PARTICLE_COUNTS[_cellx]
        for _layerx in 1:_npart_in_cell
            ix = _C2P_MAP[(_cellx - 1) * _C2P_MAP_STRIDE + _layerx]
        """
        finalise_loop_2 = "end"

        # function to get the size of the iteration set
        first_iteration_size = () -> return mesh.cell_count * get_iteration_set_size_mesh_dofs(first_cell_dat)

        # function to run to build cell to particle maps
        init_particle_map = () -> return assemble_cell_to_particle_map(cell_to_particle_map)

        init_mask = """
        # mask off the dofs that don't exist (for the non-constant case)
        if (_dofx <= $dof_count_lookup)
        """
        finalise_mask = "end"

    else
        # first loop is over particles second over dofs
        init_loop_1 = """
        # assume _ix is npart_local
        ix = _ix
        """
        finalise_loop_1 = ""

        init_loop_2 = """
        # loop over cell dofs
        _cellx = _P2C_MAP[ix, 1]
        _start_dof = 1
        _end_dof = $dof_count_lookup
        for _dofx in _start_dof:_end_dof
        """
        finalise_loop_2 = "end"

        # function to get the size of the iteration set
        first_iteration_size = () -> return particle_group.npart_local

        # Do not need the map from cells to particles in this case
        # only particles to cells
        init_particle_map = () -> return true

        init_mask = ""
        finalise_mask = ""
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
        @Const(_NPART_LOCAL),
        @Const(_NCELL_LOCAL),
        @Const(_C2DOFCOUNT_MAP),
        @Const(_C2DOF_STRIDE),
        @Const(_CELL_PARTICLE_COUNTS),
        @Const(_C2P_MAP_STRIDE),
        @Const(_C2P_MAP),
        @Const(_P2C_MAP),
        $kernel_params
    )

        ATOMIC_ADD = $(get_atomic_add(target))

        _local_ix = @index(Local)
        _group_ix = @index(Group)
        _ix = @index(Global)
        
        $init_loop_1 # init loop 1 - loop over dofs or particles

        $init_mask
        
            $(pre_kernel_launch[1])
            $pre_kernel_sync

            @inbounds begin
                $init_loop_2 # init loop 2

                    $(pre_kernel_launch[2])
                    
                    $(kernel.source)

                $finalise_loop_2 #end loop 2
            end #inbounds

        $finalise_mask
        $finalise_loop_1 # end loop 1

        $post_kernel_launch

    end
    """

    #println(kernel_func)

    l = Task(
        kernel.name * "_" * string(target) * "_DOFParticlePairLoop",
        () -> return
    )

    # Pass the kernel function to KernelAbstractions
    @eval new_kernel() = $(Meta.parse(kernel_func))
    
    ka_methods = Base.invokelatest(new_kernel)
    loop_func = Base.invokelatest(ka_methods, target.device, target.workgroup_size)
    
    #a = -1
    # function for the task
    function loop_wrapper()
        
        npart_local = particle_group.npart_local
        ncell_local = mesh.cell_count
        max_dof_count = get_max_dof_count()
        cell2particle_map = get_cell_to_particle_map(mesh, particle_group)
        
        #if a < 0
        assemble_map_if_required(cell2particle_map)
        #a = 1
        #end

        p2cell_dat = cellid_particle_dat(particle_group, mesh)
        p2cell_dat_arg = get_loop_args(npart_local, "_P2C_MAP", p2cell_dat, READ, target)[1]

        call_args = flatten([get_loop_args(npart_local, px.first, px.second[1], px.second[2], target) for px in args])
        N = first_iteration_size()

        if typeof(cell2dofcount_map) == CellDat
            cell2dofcount_arg = cell2dofcount_map.data
        else
            cell2dofcount_arg = cell2dofcount_map
        end
        
        #@show N
        #@show    npart_local
        #@show    ncell_local
        #@show    cell2dofcount_map
        #@show    max_dof_count
        #@show    cell2particle_map.cell_npart.data
        #@show    cell2particle_map.cell_children.stride
        #@show    cell2particle_map.cell_children.data
        #@show    p2cell_dat_arg
        #@show    call_args

        t0 = time_ns()
        #@profile begin
        event = loop_func(
            npart_local,
            ncell_local,
            cell2dofcount_map,
            max_dof_count,
            cell2particle_map.cell_npart.data,
            cell2particle_map.cell_children.stride,
            cell2particle_map.cell_children.data,
            p2cell_dat_arg,
            call_args...,
            ndrange=N
        )
        wait(event)
        #end
        #Profile.print()
        t1 = time_ns()
        l.runtime_inner += Float64(t1 - t0) * 1E-9
        #@show Float64(t1 - t0) * 1E-9

        # handle any post loop procedures
        for px in zip(args, call_args)
            post_loop(N, px[1].first, px[1].second[1], px[1].second[2], target, px[2])
        end

    end


    l.execute = loop_wrapper
    return l
end


function PairLoop(
    target,
    kernel,
    args_a,
    args_b
)
    return DOFParticlePairLoop(target, kernel, args_a, args_b)
end


