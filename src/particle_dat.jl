export ParticleDat, getindex, setindex
using CUDA
CUDA.allowscalar(true)


"""
Struct to hold per particle properties. e.g. Positions, charge.
"""
mutable struct ParticleDat
    ncomp::Int64
    dtype::DataType
    position::Bool
    version_id::BigInt
    data::Any
    compute_target::Any
    particle_group::ParticleGroup
    npart_local::Int64
    function ParticleDat(ncomp::Int64, dtype::DataType=Float64; position::Bool=false)
        return new(ncomp, dtype, position, BigInt(0))
    end
end


"""
Allow access to ParticleDats data using subscripts.
"""
function Base.getindex(dat::ParticleDat, key...)
    base_array = get_data_on_host(dat, dat.compute_target, (1:dat.npart_local, 1:dat.ncomp))
    base_array = base_array[key...]
    if typeof(base_array) <: SubArray
        return convert(Array{dat.dtype}, base_array)
    else
        return base_array
    end

end


"""
Allow writing data to a ParticleDat
"""
function Base.setindex!(dat::ParticleDat, value, key...)
    dat.data[key...] = value
    dat.version_id += 1
end


"""
Pretty(ish) printing of ParticleDats
"""
function Base.show(io::IO, particle_dat::ParticleDat)
    print(io, "\nnpart_local: ", particle_dat.npart_local, "\n")
    print(io, "dtype:       ", particle_dat.dtype, "\n")
    print(io, "ncomp:       ", particle_dat.ncomp, "\n")
    if isdefined(particle_dat, :compute_target)
        print(io, "target:      ", particle_dat.compute_target, "\n")
    end
    if isdefined(particle_dat, :data)
        data = particle_dat[:,:]
        for ix in 1:particle_dat.npart_local
            print(io, data[ix, :], "\n")
        end
    end
end


"""
Zero the values in a ParticleDat
"""
function zero_dat(particle_dat::ParticleDat)
    fill!(particle_dat.data, 0.0)
end


"""
Create a container to store particle data, for use in ParticleDat, for a given
compute target.
"""
function particle_data_buffer(dtype::DataType, shape::Tuple, compute_target::T) where (T<:KernelAbstractionsDevice)
    data = compute_target.ArrayType{dtype}(undef, shape)
    return data
end


"""
Initialise the inner container that stores particle data.
"""
function init_particle_data(particle_dat)
    particle_dat.data = particle_data_buffer(
        particle_dat.dtype, (0, particle_dat.ncomp), particle_dat.compute_target
    )
    particle_dat.npart_local = 0
end


"""
Copy the contents of a container to a another. e.g. when storage for a
container is reallocated.
"""
function copy_particle_data(dest, source, compute_target::T) where (T<:KernelAbstractionsDevice)
    
    size_d = size(dest)
    size_s = size(source)
    
    if (size_d[2] != size_s[2])
        error("Source and destination have a different number of columns.")
    end

    if (size_d[1] < size_s[1])
        error("Destination does not have enough rows.")
    end

    dest[1:size_s[1], :] = source[:, :]

end


"""
Ensure that the container in a ParticleDat can contain at least N particles.
"""
function grow_particle_dat(particle_dat, N)
    if (size(particle_dat.data)[1] >= N)
        return
    end
    
    # create a new buffer of the required size
    new_data = particle_data_buffer(
        particle_dat.dtype, (N, particle_dat.ncomp), particle_dat.compute_target
    )
    
    # copy the old data into the new buffer
    copy_particle_data(new_data, particle_dat.data, particle_dat.compute_target)
    
    # assign the new buffer to the particledat
    particle_dat.data = new_data

end


"""
Append the passed data onto the ParticleDat.
"""
function append_particle_data(particle_dat, data)
    N, ncomp = size(data)
    if (ncomp != particle_dat.ncomp)
        error("Missmatch between data and ParticleDat ncomp.")
    end

    if size(particle_dat.data)[1] < (particle_dat.npart_local + N)
        error("ParticleDat has insuffcient space.")
    end
    
    npart_local = particle_dat.npart_local
    new_npart_local = particle_dat.npart_local + N
    particle_dat.data[npart_local + 1: new_npart_local, :] = data[:,:]
    particle_dat.npart_local = new_npart_local
    particle_dat.version_id += 1
end


