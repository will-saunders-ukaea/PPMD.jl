export hash_primitive_type


"""
Convert a primitive type to a hex string.
"""
function hash_primitive_type(a)
    return string(parse(Int64, bitstring(a), base=2), base=16)
end
