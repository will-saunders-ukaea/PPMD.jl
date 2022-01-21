export setup_local_transfer, NeighbourExchangeRanks


using MPI


"""
Data structure to hold send/recv ranks for neighbour based transfer of
particles.
"""
struct NeighbourExchangeRanks
    # Is this initialised?
    init::Bool
    # Remote ranks this rank could recv from.
    recv_ranks::Array{Cint}
    # Remote ranks this rank could send to.
    send_ranks::Array{Cint}
    function NeighbourExchangeRanks()
        return new(false, Array{Cint}(undef, 0), Array{Cint}(undef, 0))
    end
    function NeighbourExchangeRanks(recv_ranks::Array{Cint}, send_ranks::Array{Cint})
        return new(true, recv_ranks, send_ranks)
    end
end


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
    recv_win = Win(comm, 1, Cint)
    remote_offsets = Array{Cint}(undef, N_remote_ranks)
    Cint_one = ones(Cint, 1)
    for rankx in 1:N_remote_ranks
        ranki = neighbour_ranks[rankx]
        MPI.Win_lock(MPI.LOCK_SHARED, ranki, 0, recv_win.win)
        MPI.Get_accumulate(
            view(Cint_one, 1:1),
            view(remote_offsets, rankx:rankx),
            ranki, 
            0, 
            MPI.SUM,
            recv_win.win
        )
        MPI.Win_unlock(ranki, recv_win.win)
    end
    MPI.Barrier(comm)
    recv_count = recv_win.buffer[1]
    free(recv_win)
    
    # allocate space and an MPI Window for the remote ranks
    sending_ranks_win = Win(comm, recv_count, Cint)

    # reuse this temp array to send the rank
    Cint_one[1] = rank
    # place this rank integer in the remote window
    for rankx in 1:N_remote_ranks
        ranki = neighbour_ranks[rankx]
        MPI.Win_lock(MPI.LOCK_SHARED, ranki, 0, sending_ranks_win.win)
        MPI.Put(
            view(Cint_one, 1:1),
            ranki, 
            remote_offsets[rankx], 
            sending_ranks_win.win
        )
        MPI.Win_unlock(ranki, sending_ranks_win.win)
    end
    MPI.Barrier(comm)

    sending_ranks = Array{Cint}(undef, recv_count)
    sending_ranks[:] .= sending_ranks_win.buffer[:]

    free(sending_ranks_win)
    
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
    
    # store these ranks on the particle group
    particle_group.neighbour_exchange_ranks = NeighbourExchangeRanks(
        remote_ranks,
        neighbour_ranks
    )

end


"""
Get the local ids from an array of particle ids that can not be transferred
using the neighbour transfer approach.
"""
function get_non_neighbour_ids(particle_group, particle_indices, owning_ranks)
    
    neighbour_send_ranks = particle_group.neighbour_exchange_ranks.send_ranks

    if length(neighbour_send_ranks) == 0
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


