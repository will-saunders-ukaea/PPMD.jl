export ParticleGroup, add_particles, remove_particles, getindex, initialise_particle_group_move, global_transfer_to_rank, add_particle_dat

using MPI
using DataStructures


"""
Add a ParticleDat to a ParticleGroup.
"""
function add_particle_dat(group, name, particle_dat)
    particle_dat.particle_group = group
    particle_dat.compute_target = group.compute_target
    group.particle_dats[name] = particle_dat
    init_particle_data(particle_dat)
    grow_particle_dat(particle_dat, group.npart_local)
    particle_dat.npart_local = group.npart_local
end


"""
The ParticleGroup struct contains a collection of ParticleDat instances that
define the properties of the particles in the ParticleGroup. One of these
ParticleDats should have the ``position`` field set to ``true`` to indicate
that that particular ParticleDat contains the positions of the particles.
"""
mutable struct ParticleGroup
    domain
    particle_dats::Dict
    compute_target::Any
    npart_local::Int64
    boundary_condition_task
    position_to_rank_task

    position_dat::String

    function ParticleGroup(domain, particle_dats, compute_target=false)
        new_particle_group = new(domain, OrderedDict(), compute_target, 0, false, false)
        
        # ParticleDats required intenally by the implementation
        internal_dats = Dict(
            #"_owning_rank" => ParticleDat(1, Cint),
            # Theres a bug in CUDA.jl that prevents casting to Cint in the kernel
            # https://githubmemory.com/repo/JuliaGPU/KernelAbstractions.jl/issues/254
            "_owning_rank" => ParticleDat(1, Float64),
        )
        particle_dats = merge(particle_dats, internal_dats)
        
        # ensure consistent order accross MPI ranks
        particle_dat_names = sort([dx for dx in keys(particle_dats)])
        
        # add the dats to the particle group
        for namex in particle_dat_names
            datx = particle_dats[namex]
            add_particle_dat(new_particle_group, namex, datx)
            if (datx.position)
                new_particle_group.position_dat = namex
            end
        end

        initialise_particle_group_move(new_particle_group, true)

        return new_particle_group
    end
end


"""
Initialise PBC and position to rank on ParticleGroup
"""
function initialise_particle_group_move(particle_group::ParticleGroup, re_init=false)
    
    if ((typeof(particle_group.boundary_condition_task) == Bool) || re_init)
        # Task to apply boundary condition to particles
        particle_group.boundary_condition_task = get_boundary_condition_loop(
            particle_group.domain.boundary_condition,
            particle_group
        )
    end
    
    if ((typeof(particle_group.position_to_rank_task) == Bool) || re_init)
        # Task that maps position to MPI rank
        particle_group.position_to_rank_task = get_position_to_rank_loop(particle_group)
    end

end


"""
Allow access to ParticleDats in a ParticleGroup using subscripts.
"""
function Base.getindex(group::ParticleGroup, key::String)
    return group.particle_dats[key]
end


"""
Add new particles to a ParticleGroup instance. New particle data is passed as a
Dict where the keys are the strings that name the destination ParticleDats and
the values are Arrays of the appropriate size. i.e. The number of rows
determines the number of new particles to be added and the number of columns
should be equal to the number of components of the ParticleDat.

This call should be collective across the domain communicator of the
ParticleGroup.
"""
function add_particles(group::ParticleGroup, particle_data::Dict=Dict())

    if (!isempty(particle_data))
        # Check data has consistent sizes.
        N = -1
        for dx in particle_data
            nprop, ncomp = size(dx.second)
            if (N < 0)
                N = nprop
            elseif (N != nprop)
                error("Data to initialise particles with has inconsistent numbers of rows.")
            end
            
            if (group.particle_dats[dx.first].ncomp != ncomp) 
                error("Data to initialise ParticleDat $dx.first has incorrect number of components.")
            end

        end
        
        group.npart_local += N
        for dx in group.particle_dats
            dat = dx.second

            # expand the particle dat storage to allocate space for the new
            # particles
            grow_particle_dat(dat, N + size(dat.data)[1])
            if (dx.first in keys(particle_data))
                new_data = particle_data[dx.first]
            else
                new_data = zeros(dat.dtype, (N, dat.ncomp))
            end
            append_particle_data(dat, new_data)
            @assert dat.npart_local == group.npart_local
        end
    end
    
    # Move added particles to the correct owning rank
    # This is why add_particles is collective over the particle group
    global_move(group)

end


"""
Remove particles from the ParticleGroup. Indices should be an iterable
containing the local indices of particles to be removed.

This call should be collective across the domain communicator of the
ParticleGroup.
"""
function remove_particles(group::ParticleGroup, indices)

    npart_local = group.npart_local
    indices = Set(indices)
    for ix in indices
        if ((ix < 1) || (ix>npart_local))
            error("Particle index $ix does not exist (npart_local = $npart_local)")
        end
    end
    
    N_remove = length(indices)
    npart_local = group.npart_local
    new_npart_local = npart_local - N_remove
    @assert new_npart_local >= 0
    
    # find the particles which we want to move into the newly vacated slots.
    to_move = filter(x->!(x in indices), range(new_npart_local+1, npart_local, step=1))

    # find the newly vacated indices which should be filled.
    filter!(x->x<=new_npart_local, indices)
    @assert length(to_move) == length(indices)
    
    # copy the particle data from the end of the particledat into the newly
    # vacated slots.
    for dat in group.particle_dats
        for ix in zip(indices, to_move)
            dat.second.data[ix[1], :] = dat.second.data[ix[2], :]
        end
        dat.second.npart_local = new_npart_local
        dat.second.version_id += 1
    end
    group.npart_local = new_npart_local

end


"""
Loop over an Array{Cint} once and collect in a dictionary the locations of each
element whilst using the elements as keys.
"""
function map_elements_to_location(dest_rank::Array{Cint}, indices_to_send::Array{Int64})
    
    m = OrderedDict{Cint, Stack{Int64}}()
    for ix in 1:length(dest_rank)
        keyx = dest_rank[ix]
        if !(keyx in keys(m))
            m[keyx] = Stack{Int64}()
        end
        push!(m[keyx], indices_to_send[ix])
    end

    return m
end


"""
Number of bytes required per particle in ParticleGroup
"""
function sizeof_particle(particle_group::ParticleGroup)
    s = 0
    for datx in particle_group.particle_dats
        s += sizeof(datx.second.dtype) * datx.second.ncomp
    end
    return s
end


"""
Pack particle data for sending as Cchar for global move
"""
function pack_particle_dats(particle_group, rank_to_indices_map, send_buffer)

    bytes_per_particle = sizeof_particle(particle_group)
    column_index = 1
    for rankx in keys(rank_to_indices_map)
        for particle in rank_to_indices_map[rankx]
            row_index = 1
            for datx in particle_group.particle_dats
                ncomp = datx.second.ncomp
                stride = ncomp * sizeof(datx.second.dtype)
                send_buffer[row_index:row_index+stride-1, column_index] .= 
                    reinterpret(Cchar, datx.second[particle, 1:ncomp])
                row_index += stride
            end
            column_index += 1
        end
    end
end


"""
Unpack a Cchar buffer into ParticleDats.
"""
function unpack_particle_dats(particle_group, recv_count, recv_buffer)
    
    old_npart_local = particle_group.npart_local
    particle_group.npart_local += recv_count
    
    row_index = 1
    for dat in particle_group.particle_dats
        # realloc if needed
        grow_particle_dat(dat.second, old_npart_local + recv_count)

        # unpack recvd data
        ncomp = dat.second.ncomp
        stride = ncomp * sizeof(dat.second.dtype)       
        new_data = reinterpret(
                dat.second.dtype, 
                recv_buffer[row_index:row_index+stride-1, :]
            )

        new_data = transpose(new_data)
        
        append_particle_data(dat.second, new_data)
        @assert dat.second.npart_local == particle_group.npart_local
        row_index += stride
    end

end


"""
Transfer particle ownership to ranks specified in the _owning_rank ParticleDat.
Does not require any restrictions on the destination ranks, i.e. an any-to-any
transfer.
"""
function global_transfer_to_rank(particle_group)

    # Get the owning ranks
    owning_ranks = particle_group["_owning_rank"][:, 1]

    # work around the CUDA.jl issue for casting to int in kernels
    if (particle_group["_owning_rank"].dtype != Cint)
        owning_ranks = convert(Array{Cint}, round.(owning_ranks))
    end
    
    comm = particle_group.domain.comm
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    @assert length(filter((x) -> x < 0, owning_ranks)) == 0
    @assert length(filter((x) -> x > size, owning_ranks)) == 0
    
    # Find the indices of particles which should be transferred
    indices_to_send = findall((x) -> x != rank, owning_ranks)
    # Get the corresponding destination ranks
    dest_ranks = owning_ranks[indices_to_send]

    rank_to_indices_map = map_elements_to_location(dest_ranks, indices_to_send)
    num_remote_ranks = length(rank_to_indices_map)
    remote_ranks = [rx for rx in keys(rank_to_indices_map)]

    # Buffer for accumulation window
    recv_counts = zeros(Cint, 1)
    # Create MPI Window for acculation of counts
    recv_win = MPI.Win_create(recv_counts, comm)
    # Create local buffer for remote offsets
    send_offsets = zeros(Cint, num_remote_ranks)
    send_counts = zeros(Cint, num_remote_ranks)

    # loop over ranks to transfer to
    send_total_count = 0
    for rankx in 1:num_remote_ranks
        ranki = remote_ranks[rankx]
        # lock remote rank for window in MPI_LOCK_SHARED
        MPI.Win_lock(MPI.LOCK_SHARED, ranki, 0, recv_win)
        # Get_accumulate counts (send this ranks count to go to remote rank and
        # recv the remote offset)
        send_count = length(rank_to_indices_map[ranki])
        send_total_count += send_count
        send_counts[rankx] = send_count
        MPI.Get_accumulate(
            view(send_counts, rankx:rankx),
            view(send_offsets, rankx:rankx),
            ranki, 
            0, 
            MPI.SUM,
            recv_win
        )
        # unlock remote rank on window
        MPI.Win_unlock(ranki, recv_win)
    end
    # The Barrier for this remote access is after this allocation purely
    # in the hope that the allocation/pack can happen whilst the Get_accumulate
    # occurs.
    
    # Pack the data to send
    bytes_per_particle = sizeof_particle(particle_group)
    send_buffer = Array{Cchar}(undef, (bytes_per_particle, send_total_count))
    pack_particle_dats(particle_group, rank_to_indices_map, send_buffer)

    # Barrier for the send counts/acculation access
    MPI.Barrier(comm)
    # Allocate space for the data we are about to recv
    recv_count = recv_counts[1]
    #recv_buffer = Array{Cchar}(undef, (bytes_per_particle, recv_count))
    recv_buffer = zeros(Cchar, (bytes_per_particle, recv_count))
    recv_data_win = MPI.Win_create(recv_buffer, comm)

    # loop over ranks and Put the data in remote buffers
    send_offset = 1
    for rankx in 1:num_remote_ranks
        ranki = remote_ranks[rankx]
        # lock remote rank for window in MPI_LOCK_SHARED
        MPI.Win_lock(MPI.LOCK_SHARED, ranki, 0, recv_data_win)
        # Get_accumulate counts (send this ranks count to go to remote rank and
        # recv the remote offset)
        send_count = send_counts[rankx]
        
        # deduce the target offset from the bytes per particle and the offset
        target_disp = bytes_per_particle * send_offsets[rankx]

        # put the packed data to send in the remote buffer
        MPI.Put(
            view(send_buffer, :, send_offset:send_offset+send_count-1),
            ranki, 
            target_disp, 
            recv_data_win
        )
        send_offset += send_count

        # unlock remote rank on window
        MPI.Win_unlock(ranki, recv_data_win)
    end
    

    # remove the particles that were sent
    remove_particles(particle_group, indices_to_send)

    # Barrier for the Put operations
    MPI.Barrier(comm)
    
    # Unpack the recvd data
    unpack_particle_dats(particle_group, recv_count, recv_buffer)
    
    # Free accumulation window
    MPI.free(recv_win)
    MPI.free(recv_data_win)

end


"""
Send particles to correct owning rank. Is suitable for a global move.
"""
function global_move(particle_group)
    
    # Execute the ParticleLoop/Task that applies the boundary conditions
    execute(particle_group.boundary_condition_task)
    # Map the particle positions to MPI ranks
    execute(particle_group.position_to_rank_task)
    # Transfer ownership to the new ranks
    global_transfer_to_rank(particle_group)

end




