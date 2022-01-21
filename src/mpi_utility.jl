export alloc_mem, free_mem, MPIArray, free

using MPI


"""
Allocate memory using MPI_Alloc_mem which is required for portable MPI.
"""
function alloc_mem(size, dtype)

    ptr_ref = Ref{Ptr{Cchar}}()
    size_bytes = size * sizeof(dtype)

    # int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr)
    MPI.@mpichk ccall(
        (:MPI_Alloc_mem, libmpi), 
        Cint,
        (MPI.MPI_Aint, MPI.MPI_Info, Ref{Ptr{Cchar}}),
        MPI.MPI_Aint(size_bytes), 
        MPI.Info(), 
        ptr_ref
    )
    
    ptr = Ptr{Cchar}(ptr_ref[])
    ptr = convert(Ptr{dtype}, ptr)

    return ptr
end


"""
Free memory allocated with MPI_Alloc_mem
"""
function free_mem(ptr)

    ptr = convert(Ptr{Cchar}, ptr)

    # int MPI_Free_mem(void *base)
    MPI.@mpichk ccall(
        (:MPI_Free_mem, libmpi),
        Cint,
        (Ptr{Cchar},),
        ptr
    )

end


"""
Allocate an Array with MPI and wrap in an Array.
"""
mutable struct MPIArray
    size
    dtype
    array
    ptr
    function MPIArray(size, dtype)
        ptr = alloc_mem(size, dtype)
        arr = unsafe_wrap(Array{dtype}, ptr, size)
        return new(size, dtype, arr, ptr)
    end
end


"""
Free an MPIArray.
"""
function free(arr::MPIArray)
    arr.size = 0
    arr.array = zeros(arr.dtype, 0)
    free_mem(arr.ptr)
    arr.ptr = Ptr{arr.dtype}(0)
end


"""
Wrapper around an MPI Win. Must be created collectively on the MPI
Communicator.
"""
mutable struct Win
    mpiarray::MPIArray
    buffer::Array
    comm::MPI.Comm
    win::MPI.Win
    function Win(comm, size, dtype)
        mpiarray = MPIArray(size, dtype)
        buffer = mpiarray.array
        buffer[:] .= 0
        win = MPI.Win_create(buffer, comm)
        return new(mpiarray, buffer, comm, win)
    end
end


"""
Free a Win object. Must be called collectively on the MPI communicator.
"""
function free(win::Win)
    MPI.free(win.win)
    free(win.mpiarray)
end



















