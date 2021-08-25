export ParticleDat, particle_data_buffer, init_particle_data, copy_particle_data, grow_particle_dat, append_particle_data


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


function particle_data_buffer(dtype::DataType, shape::Tuple, compute_target::T) where (T<:KernelAbstractionsDevice)
    data = compute_target.ArrayType{dtype}(undef, shape)
    return data
end


function init_particle_data(particle_dat)
    particle_dat.data = particle_data_buffer(
        particle_dat.dtype, (0, particle_dat.ncomp), particle_dat.compute_target
    )
    particle_dat.npart_local = 0
end


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



function grow_particle_dat(particle_dat, N)
    if (size(particle_dat.data)[1] >= N)
        return
    end

    new_data = particle_data_buffer(
        particle_dat.dtype, (N, particle_dat.ncomp), particle_dat.compute_target
    )

    particle_dat.data = new_data

end


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

    particle_dat.data[npart_local + 1: new_npart_local, :] = data
    particle_dat.npart_local = new_npart_local
    particle_dat.version_id += 1
end
