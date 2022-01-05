export print_profile, set_profiling_value, increment_profiling_value, reset_profile, get_profile_dict


"""
Hold profiling data.
"""
mutable struct ProfileDat
    data::Dict{String, Dict{String, Any}}
    function ProfileDat()
        return new(Dict())
    end
end


"""
Set values in a ProfileDat like PROFILE["A", "B"] = 42.
"""
function Base.getindex(dat::ProfileDat, key1, key2)
    return get(get(dat.data, key1, Dict()), key2, undef)
end


"""
Get values in a ProfileDat like PROFILE["A", "B"].
"""
function Base.setindex!(dat::ProfileDat, value, key1, key2)

    l2_dict = get(dat.data, key1, undef)
    if l2_dict == undef
        l2_dict = Dict{String, Any}()
        l2_dict[key2] = value
        dat.data[key1] = l2_dict
    else
        l2_dict[key2] = value
    end

end


"""
Global to hold global profiling data.
"""
PROFILE = ProfileDat()


"""
Set a value in the global profiling data structure.
"""
function set_profiling_value(key1, key2, value)
    PROFILE[key1, key2] = value
end


"""
Increment a value in the global profiling data structure.
"""
function increment_profiling_value(key1, key2, value)
    current_value = PROFILE[key1, key2]
    if current_value == undef
        PROFILE[key1, key2] = value
    else
        PROFILE[key1, key2] += value
    end

end


"""
Function to print a profile using show.
"""
function Base.show(io::IO, dat::ProfileDat)
    
    print(io, "\n")
    for keyx in keys(dat.data)
        print(io, keyx, ":\n")
        for inner_keyx in keys(dat.data[keyx])
            print(io, "  $inner_keyx:  ", dat.data[keyx][inner_keyx], "\n")
        end
    end

end


"""
Helper to print the profile.
"""
function print_profile()
    @show PROFILE
end


"""
Function to reset global profiling data collection.
"""
function reset_profile()
    global PROFILE = ProfileDat()
    set_profiling_value("MPI", "MPI_COMM_WORLD_rank", MPI.Comm_rank(MPI.COMM_WORLD))
    set_profiling_value("MPI", "MPI_COMM_WORLD_size", MPI.Comm_size(MPI.COMM_WORLD))
end


"""
Get the profiling data dict
"""
function get_profile_dict()
    return PROFILE.data
end


