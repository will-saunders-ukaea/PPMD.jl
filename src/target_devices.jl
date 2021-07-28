
using KernelAbstractions
using CUDAKernels

struct KADevice
    device::Any
    workgroup_size::Int64
end

KACPU = CPU
KACUDADevice = CUDADevice



