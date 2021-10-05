export READ, WRITE, INC, DirectAccess
struct AccessType
    write::Bool
end


"Access descriptor for read-only access"
READ = AccessType(false)

"Access descriptor for write access"
WRITE = AccessType(true)

"Access descriptor for increment access"
INC = AccessType(true)


struct DirectAccess
    dat
end

