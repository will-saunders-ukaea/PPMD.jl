#using Debugger
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
                    @show pos, bounds
                    return false
                end
            end
            
        end
        
    end

    return true
end


##@testset "PBC global_move $spec" for spec in Iterators.product((KACPU(), KACUDADevice()), (1, 2, 3), (global_move, ), (0, 1))
##@testset "PBC global_move $spec" for spec in Iterators.product((KACUDADevice(),), (1, 2, 3), (global_move,), (0, 1))
#@testset "PBC global_move $spec" for spec in Iterators.product((KACUDADevice(),), (1,), (global_move,), (1,))
#    
#
#    @show spec
#
#    target_device = spec[1]
#    ndim = spec[2]
#    move_method = spec[3]
#    stencil_width = spec[4]
#
#    N = 20
#
#    if ndim == 1
#        extents = [2.0,]
#    elseif ndim == 2
#        extents = [2.0, 3.0]
#    elseif ndim == 3
#        extents = [2.0, 3.0, 4.0]
#    end
#
#
#    boundary_condition = FullyPeroidicBoundary()
#    domain = StructuredCartesianDomain(boundary_condition, extents)
#
#    A = ParticleGroup(
#        domain,
#        Dict(
#             "P" => ParticleDat(ndim, position=true),
#             "ID" => ParticleDat(1, Int64),
#             "P_INITIAL" => ParticleDat(ndim),
#             "V" => ParticleDat(ndim),
#        ),
#        target_device
#    )
#
#    neigbour_ranks = get_neighbour_ranks(domain, stencil_width)
#    setup_local_transfer(A, neigbour_ranks)
#    
#    size = MPI.Comm_size(domain.comm)
#    rank = MPI.Comm_rank(domain.comm)
#    
#    rng = MersenneTwister(1234*(rank+1))
#
#    dt = 0.1
#    v_max = Float64(maximum(extents) / (0.5 * dt))
#    v_initial = v_max .- rand(rng, Float64, N, ndim) * 2.0 * v_max
#    p_initial = rand_within_extents(N, domain.extent)
#
#    add_particles(
#        A,
#        Dict(
#             "P" => p_initial,
#             "P_INITIAL" => p_initial,
#             "ID" => reshape([ix for ix in 1:N], (N, 1)) .+ N*rank,
#             "V" => v_initial,
#        )
#    )
#
#    
#    advection = ParticleLoop(
#        target_device,
#        Kernel(
#            "advection_kernel",
#            """
#            for dx in 1:$ndim
#                P[ix, dx] += V[ix, dx] * $dt
#            end
#            """
#        ),
#        Dict(
#             "P" => (A["P"], WRITE),
#             "V" => (A["V"], READ),
#        )
#    )
#    
#    move_method(A)
#
#    dims, periods, coords = MPI.Cart_get(domain.comm)
#    bounds = zeros(Float64, ndim, 2)
#    for dx in 1:ndim
#        width = extents[dx] / dims[dx]
#        bounds[dx, 1] = coords[dx] * width
#        bounds[dx, 2] = (coords[dx] + 1) * width
#    end
#
#
#    tol = 1E-11
#    
#    #Base.GC.enable(false)
#    #@show rank
#    for stepx in 1:200
#        @show stepx, getpid(), spec
#        execute(advection)
#        MPI.Barrier(domain.comm)
#        println("advection performed")
#        #@show A["P"]
#        #@show A["_owning_rank"]
#        move_method(A)
#        #@show A["P"]
#        #@show A["_owning_rank"]
#        #@show stepx
#        #
#
#        #for px in 1:A.npart_local
#        #    @test check_in_bounds(A["P"][px, :], bounds, tol)
#        #    correct_position = vector_fmod(A["P_INITIAL"][px, :] + stepx * dt * A["V"][px, :], extents)
#        #    for dx in 1:ndim
#        #
#        #        #if !((norm(A["P"][px, dx] - correct_position[dx], Inf) < tol) || (abs(norm(A["P"][px, dx] - correct_position[dx], Inf) - extents[dx]) < tol))
#        #        #    @show norm(A["P"][px, dx] - correct_position[dx], Inf)
#        #        #    @show abs(norm(A["P"][px, dx] - correct_position[dx], Inf) - extents[dx])
#        #        #end
#        #        @test (norm(A["P"][px, dx] - correct_position[dx], Inf) < tol) || (abs(norm(A["P"][px, dx] - correct_position[dx], Inf) - extents[dx]) < tol)
#        #    end
#        #end
#    end
#    
#    #Base.GC.enable(true)
#    free(A)
#    free(domain)
#
#end




function foo()

    spec = (KACUDADevice(), 1, global_move, 1)

    @show spec

    target_device = spec[1]
    ndim = spec[2]
    move_method = spec[3]
    stencil_width = spec[4]

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
    neigbour_ranks = get_neighbour_ranks(domain, stencil_width)
    setup_local_transfer(A, neigbour_ranks)
    
    size = MPI.Comm_size(domain.comm)
    rank = MPI.Comm_rank(domain.comm)
    
    rng = MersenneTwister(1234*(rank+1))

    dt = 0.1
    v_max = Float64(maximum(extents) / (0.5 * dt))
    v_initial = v_max .- rand(rng, Float64, N, ndim) * 2.0 * v_max
    p_initial = rand_within_extents(N, domain.extent)

    add_particles(
        A,
        Dict(
             "P" => p_initial,
             "P_INITIAL" => p_initial,
             "ID" => reshape([ix for ix in 1:N], (N, 1)) .+ N*rank,
             "V" => v_initial,
        )
    )
    
    return
    
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
    
    move_method(A)

    dims, periods, coords = MPI.Cart_get(domain.comm)
    bounds = zeros(Float64, ndim, 2)
    for dx in 1:ndim
        width = extents[dx] / dims[dx]
        bounds[dx, 1] = coords[dx] * width
        bounds[dx, 2] = (coords[dx] + 1) * width
    end


    tol = 1E-11
    
    #Base.GC.enable(false)
    #@show rank
    for stepx in 1:200
        @show stepx, getpid(), spec
        execute(advection)
        MPI.Barrier(domain.comm)
        println("advection performed")
        #@show A["P"]
        #@show A["_owning_rank"]
        move_method(A)
        #@show A["P"]
        #@show A["_owning_rank"]
        #@show stepx
        #

        #for px in 1:A.npart_local
        #    @test check_in_bounds(A["P"][px, :], bounds, tol)
        #    correct_position = vector_fmod(A["P_INITIAL"][px, :] + stepx * dt * A["V"][px, :], extents)
        #    for dx in 1:ndim
        #
        #        #if !((norm(A["P"][px, dx] - correct_position[dx], Inf) < tol) || (abs(norm(A["P"][px, dx] - correct_position[dx], Inf) - extents[dx]) < tol))
        #        #    @show norm(A["P"][px, dx] - correct_position[dx], Inf)
        #        #    @show abs(norm(A["P"][px, dx] - correct_position[dx], Inf) - extents[dx])
        #        #end
        #        @test (norm(A["P"][px, dx] - correct_position[dx], Inf) < tol) || (abs(norm(A["P"][px, dx] - correct_position[dx], Inf) - extents[dx]) < tol)
        #    end
        #end
    end
    
    #Base.GC.enable(true)
    free(A)
    free(domain)

end


foo()
