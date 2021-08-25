
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
        return new(CUDADevice(), workgroup_size, CuArray)
    end
end


