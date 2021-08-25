export ParticleGroup, add_particles, remove_particles


"""
Add a ParticleDat to a ParticleGroup.
"""
function add_particle_dat(group, name, particle_dat)
    particle_dat.particle_group = group
    particle_dat.compute_target = group.compute_target
    group.particle_dats[name] = particle_dat
    init_particle_data(particle_dat)
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

    position_dat::String
    function ParticleGroup(domain, particle_dats, compute_target=false)
        new_particle_group = new(domain, Dict(), compute_target, 0)
        for datx in particle_dats
            add_particle_dat(new_particle_group, datx.first, datx.second)
            if (datx.second.position)
                new_particle_group.position_dat = datx.first
            end
        end
        return new_particle_group
    end
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
function add_particles(group::ParticleGroup, particle_data::Dict)
    
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
    
    # expand the particle dat storage to allocate space for the new particles
    group.npart_local += N
    for dx in group.particle_dats
        dat = dx.second
        grow_particle_dat(dat, N + size(dat.data)[1])
        if (dx.first in keys(particle_data))
            new_data = particle_data[dx.first]
        else
            new_data = zeros(dat.dtype, (N, dat.ncomp))
        end
        append_particle_data(dat, new_data)
        println(dat.npart_local, " ", group.npart_local)
        @assert dat.npart_local == group.npart_local
    end

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
    filter!(x->x<new_npart_local, indices)

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