
using KernelAbstractions
using CUDA
using CUDAKernels


"Base type for targets built on top of KernelAbstractions"
abstract type KernelAbstractionsDevice end


struct KACPU <: KernelAbstractionsDevice
    device
    workgroup_size::Int64
    ArrayType
    function KACPU(workgroup_size=32)
        return new(CPU(), workgroup_size, Array)
    end
end


struct KACUDADevice <: KernelAbstractionsDevice
    device
    workgroup_size::Int64
    ArrayType
    function KACUDADevice(workgroup_size=32)
        if CUDA.functional()
            return new(CUDADevice(), workgroup_size, CuArray)
        else
            println("Warning CUDA.functional returned false, using CPU.")
            return KACPU()
        end
    end
end


function get_data_on_host(particle_dat, compute_target::T) where (T <: KACPU)
    return SubArray(particle_dat.data, (1:particle_dat.npart_local, 1:particle_dat.ncomp))
end


function get_data_on_host(particle_dat, compute_target::T) where (T <: KACUDADevice)
    host_data = convert(Array{particle_dat.dtype}, particle_dat.data)
    return host_data[1:particle_dat.npart_local, 1:particle_dat.ncomp]
end
