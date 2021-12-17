export ParticleGroupVTK, write

using WriteVTK
using LazyGrids
using MPI


"""
Stuct to write particle data to files of the form filename_<rank>_<step>.vtp.
Paraview should automatically detect the step as a time series. Use the
Paraview GroupDataset filter to combine in parallel. Timesteps are written
using the write function. Only ParticleDats that exist on construction of an
instance of this struct are written. For example

A = ParticleGroup(...)

vtk_output = ParticleGroupVTK("foo", A)

# modify/create particle data

# write particle data snapshot

write(vtk_output)

"""
mutable struct ParticleGroupVTK
    filename
    particle_group
    rank
    step
    particle_dats
    function ParticleGroupVTK(filename, particle_group)
        rank = MPI.Comm_rank(particle_group.domain.comm)
        particle_dats = keys(particle_group.particle_dats)
        mkpath(dirname(filename))
        return new(filename, particle_group, rank, 0, particle_dats)
    end
end


"""
Write a vtp file per rank for the current particle state.
"""
function Base.write(pgvtk::ParticleGroupVTK)
    time_start = time()

    filename = "$(pgvtk.filename)_$(pgvtk.rank)_$(pgvtk.step)"
    pgvtk.step += 1
    
    # Create a unstructured grid with vertex per particle.
    npart_local = pgvtk.particle_group.npart_local

    vtkfile = vtk_grid(
        filename,
        transpose(pgvtk.particle_group[pgvtk.particle_group.position_dat][:, :]),
        [
            MeshCell(PolyData.Verts(), [i for i in 1:npart_local]),
        ]
    )
    
    # Write the ParticleDat data to each vertex.
    for datx in pgvtk.particle_dats
        vtkfile[datx, VTKPointData()] = transpose(pgvtk.particle_group[datx][:,:])
    end
    
    outfile = vtk_save(vtkfile)
    
    name = "ParticleGroupVTK-" * pgvtk.filename
    increment_profiling_value(name, "time" , time() - time_start)
    increment_profiling_value(name, "count" , 1)

    return outfile
end





