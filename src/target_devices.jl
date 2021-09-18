
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


function get_array_on_host(array::T) where T<: CuArray
    return convert(Array{eltype(array)}, array)
end


function get_array_on_host(array::T) where T<: Array
    return array
end


function get_data_on_host(dat, compute_target::T, shape) where (T <: KACPU)
    return SubArray(dat.data, shape)
end


function get_data_on_host(dat, compute_target::T, shape) where (T <: KACUDADevice)
    host_data = convert(Array{dat.dtype}, dat.data)
    return host_data[shape...]
end
