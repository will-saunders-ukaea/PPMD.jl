using MPI
using PPMD
using Test
using LinearAlgebra
using Random

function vector_fmod(a, b)
    
    o = similar(a)
    for dx in 1:length(a)
        o[dx] = a[dx] % b[dx]
    end

    return o
end


function check_in_bounds(pos, bounds, tol=1E-12)
    
    ndim = size(bounds)[1]
    for dx in 1:ndim
        if !(pos[dx] >= bounds[dx, 1] && pos[dx] <= bounds[dx, 2])
            if !(abs(pos[dx] - bounds[dx, 1]) < tol)
                if !(abs(pos[dx] - bounds[dx, 2]) < tol)
                    return false
                end
            end
            
        end
        
    end

    return true
end


@testset "PBC global_move $spec" for spec in Iterators.product((KACPU(), KACUDADevice()), (1, 2, 3))

    target_device = spec[1]
    ndim = spec[2]

    N = 20

    if ndim == 1
        extents = [2.0,]
    elseif ndim == 2
        extents = [2.0, 3.0]
    elseif ndim == 3
        extents = [2.0, 3.0, 4.0]
    end


    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    A = ParticleGroup(
        domain,
        Dict(
             "P" => ParticleDat(ndim, position=true),
             "ID" => ParticleDat(1, Int64),
             "P_INITIAL" => ParticleDat(ndim),
             "V" => ParticleDat(ndim),
        ),
        target_device
    )
    
    size = MPI.Comm_size(domain.comm)
    rank = MPI.Comm_rank(domain.comm)
    
    rng = MersenneTwister(1234*(rank+1))

    v_max = 4.0
    v_initial = v_max .- rand(rng, Float64, N, ndim) * 2.0 * v_max
    p_initial = rand_within_extents(N, domain.extent)
    dt = 0.1

    add_particles(
        A,
        Dict(
             "P" => p_initial,
             "P_INITIAL" => p_initial,
             "ID" => reshape([ix for ix in 1:N], (N, 1)) .+ N*rank,
             "V" => v_initial,
        )
    )

    
    advection = ParticleLoop(
        target_device,
        Kernel(
            "advection_kernel",
            """
            for dx in 1:$ndim
                P[ix, dx] += V[ix, dx] * $dt
            end
            """
        ),
        Dict(
             "P" => (A["P"], WRITE),
             "V" => (A["V"], READ),
        )
    )
    
    global_move(A)

    dims, periods, coords = MPI.Cart_get(domain.comm)
    bounds = zeros(Float64, ndim, 2)
    for dx in 1:ndim
        width = extents[dx] / dims[dx]
        bounds[dx, 1] = coords[dx] * width
        bounds[dx, 2] = (coords[dx] + 1) * width
    end


    tol = 1E-12

    for stepx in 1:1000
        execute(advection)
        global_move(A)
        for px in 1:A.npart_local
            @test check_in_bounds(A["P"][px, :], bounds, tol)
            correct_position = vector_fmod(A["P_INITIAL"][px, :] + stepx * dt * A["V"][px, :], extents)
            for dx in 1:ndim
                @test (norm(A["P"][px, dx] - correct_position[dx], Inf) < tol) || (abs(norm(A["P"][px, dx] - correct_position[dx], Inf) - extents[dx]) < tol)
            end
        end
    end



end
