export READ, WRITE, INC, DirectAccessT, DirectArray, DirectAccess
struct AccessType
    write::Bool
end


"Access descriptor for read-only access"
READ = AccessType(false)

"Access descriptor for write access"
WRITE = AccessType(true)

"Access descriptor for increment access"
INC = AccessType(true)


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

