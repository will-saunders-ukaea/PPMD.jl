export setup_local_transfer


using MPI


"""
Exchange the remote MPI ranks that can send to this MPI rank. Collective on the
communicator.
"""
function exchange_neighbour_ranks(particle_group, neighbour_ranks::Array{Cint})
    
    comm = particle_group.domain.comm
    rank = MPI.Comm_rank(comm)
    N_remote_ranks = length(neighbour_ranks)

    # for each remote rank increment the remote counter to find the remote offset
    # for this rank to place its rank in the remote buffer
    recv_counts = zeros(Cint, 1)
    recv_win = MPI.Win_create(recv_counts, comm)
    remote_offsets = Array{Cint}(undef, N_remote_ranks)
    Cint_one = ones(Cint, 1)
    for rankx in 1:N_remote_ranks
        ranki = neighbour_ranks[rankx]
        MPI.Win_lock(MPI.LOCK_SHARED, ranki, 0, recv_win)
        MPI.Get_accumulate(
            view(Cint_one, 1:1),
            view(remote_offsets, rankx:rankx),
            ranki, 
            0, 
            MPI.SUM,
            recv_win
        )
        MPI.Win_unlock(ranki, recv_win)
    end
    MPI.Barrier(comm)
    MPI.free(recv_win)
    
    # allocate space and an MPI Window for the remote ranks
    sending_ranks = Array{Cint}(undef, recv_counts[1])
    sending_ranks_win = MPI.Win_create(sending_ranks, comm)

    # reuse this temp array to send the rank
    Cint_one[1] = rank
    # place this rank integer in the remote window
    for rankx in 1:N_remote_ranks
        ranki = neighbour_ranks[rankx]
        MPI.Win_lock(MPI.LOCK_SHARED, ranki, 0, sending_ranks_win)
        MPI.Put(
            view(Cint_one, 1:1),
            ranki, 
            remote_offsets[rankx], 
            sending_ranks_win
        )
        MPI.Win_unlock(ranki, sending_ranks_win)
    end
    MPI.Barrier(comm)
    MPI.free(sending_ranks_win)
    
    return sending_ranks
end


"""
Setup the data structures to support a transfer of particle ownership to a MPI
rank that is considered a neighbour of the sending rank. Collective on the
communicator.
"""
function setup_local_transfer(particle_group, neighbour_ranks::Array{Cint})
    
    # find the remote ranks that may send to this rank.
    remote_ranks = exchange_neighbour_ranks(particle_group, neighbour_ranks)
    # ranks this rank could recv from
    particle_group.maps["_local_exchange_recv_ranks"] = remote_ranks
    # ranks this rank could send to
    particle_group.maps["_local_exchange_send_ranks"] = neighbour_ranks
    particle_group.maps["_local_exchange_init"] = true

end


"""
Get the local ids from an array of particle ids that can not be transferred
using the neighbour transfer approach.
"""
function get_non_neighbour_ids(particle_group, particle_indices, owning_ranks)
    
    neighbour_send_ranks = get(particle_group.maps, "_local_exchange_send_ranks", nothing)
    if neighbour_send_ranks == nothing
        return particle_indices
    end
    
    npart_local = particle_group.npart_local
    
    N_ids = length(particle_indices)
    ids = Array{Int64}(undef, 0)

    for pid in particle_indices
        rank = owning_ranks[pid, 1]
        if !(rank in neighbour_send_ranks)
            push!(ids, pid)
        end
    end
    
    return ids
end


