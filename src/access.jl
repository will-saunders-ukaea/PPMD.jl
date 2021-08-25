
struct AccessType
    write::Bool
end


"Access descriptor for read-only access"
READ = AccessType(false)

"Access descriptor for write access"
WRITE = AccessType(true)



