using Test
using ChemistryFeaturization
using Serialization

@testset "formation energy example" begin
    include(
        joinpath(@__DIR__, "..", "examples", "1_formation_energy", "formation_energy.jl"),
    )
    @test !ismissing(
        train_formation_energy(
            num_pts = 10,
            num_epochs = 1,
            data_dir = joinpath(@__DIR__, "test_data/"),
            verbose = false,
        ),
    )
end
