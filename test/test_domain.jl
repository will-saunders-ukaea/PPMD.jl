using PPMD
using Test

@testset "domain construction" begin
    
    extents = (2.0, 3.0, 4.0)
    boundary_condition = FullyPeroidicBoundary()
    domain = StructuredCartesianDomain(boundary_condition, extents)

    @test domain.ndim == 3
    @test length(domain.extent) == 3
    @test domain.extent[1] == 2.0
    @test domain.extent[2] == 3.0
    @test domain.extent[3] == 4.0
    @test domain.boundary_condition == boundary_condition

end


