using AtomicGraphNets
using Test

@testset "layer_tests" begin
    include("layer_tests.jl")
end

@testset "model_tests" begin
    include("model_tests.jl")
end
