export READ, WRITE, INC, INC_ZERO, DirectAccessT, DirectArray, DirectAccess
struct AccessType
    write::Bool
    name::String
end


"Access descriptor for read-only access"
READ = AccessType(false, "READ")

"Access descriptor for write access"
WRITE = AccessType(true, "WRITE")

"Access descriptor for increment access"
INC = AccessType(true, "INC")

"Access descriptor for increment access with zeroing"
INC_ZERO = AccessType(true, "INC_ZERO")


abstract type DirectAccessT end

struct DirectAccess <: DirectAccessT
    dat
end


struct DataWrapper
    data
    function DataWrapper(array, compute_target)
        return new(convert(compute_target.ArrayType, array))
    end
end


struct DirectArray <: DirectAccessT
    dat
    function DirectArray(array, compute_target)
        return new(DataWrapper(array, compute_target))
    end
end

