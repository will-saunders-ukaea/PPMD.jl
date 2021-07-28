module jPPMD

export ParticleLoop, execute, READ, WRITE

include("particle_loop.jl")
include("access.jl")


function execute(loop)
    return loop()
end



end # module
