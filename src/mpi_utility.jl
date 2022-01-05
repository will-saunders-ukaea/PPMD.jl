

using MPI


"""
Wrapper around an MPI Win. Must be created collectively on the MPI
Communicator.
"""
mutable struct Win
    buffer::Array
    comm::MPI.Comm
    win::MPI.Win
    function Win(comm, size, dtype)
        # TODO potentially not portable as memory not allocated with MPI_Alloc_mem
        buffer = zeros(dtype, size)
        win = MPI.Win_create(buffer, comm)
        return new(buffer, comm, win)
    end
end


"""
Free a Win object. Must be called collectively on the MPI communicator.
"""
function free(win::Win)
    MPI.free(win.win)
end
