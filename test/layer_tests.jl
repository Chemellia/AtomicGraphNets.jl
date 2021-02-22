using Test
using ChemistryFeaturization
using SimpleWeightedGraphs

include("../src/layers.jl")
using .layers: AGNConv, AGNPool

@testset "AGNConv" begin
    # create simple line graph, populate it with feature of all ones
    adjmat = Float32.([0 1 0; 1 0 1; 0 1 0])
    dummyfzn = [AtomFeat(:feat, [0]) for i in 1:4]
    ag = AtomGraph(SimpleWeightedGraph{Int32, Float32}(adjmat), ["C", "C", "C"], ones(Float32, 4,3), dummyfzn)
    # create a conv layer, initialize weights with ones
    l = AGNConv(4=>4, initW=ones)

    # test output looks as expected
    output_fea = l(ag).features
    @test output_fea[:,1]==output_fea[:,3]
    @test isapprox(output_fea[:,1].+output_fea[:,3], .-output_fea[:,2])

    # and now for a loop
    adjmat = Float32.([0 1 1; 1 0 1; 1 1 0])
    ag = AtomGraph(SimpleWeightedGraph{Int32, Float32}(adjmat), ["C", "C", "C"], ones(Float32, 4,3), dummyfzn)
    l = AGNConv(4=>4, initW=ones)

    @test all(isapprox.(l(ag).features, zero(Float32)))
end

@testset "pooling" begin
    # keep our little line graph, but give it more features
    adjmat = Float32.([0 1 0; 1 0 1; 0 1 0])
    feat = ones(Float32, 50,3)
    dummyfzn = [AtomFeat(:feat, [0]) for i in 1:50]
    ag = AtomGraph(SimpleWeightedGraph{Int32, Float32}(adjmat), ["C", "C", "C"], feat, dummyfzn)

    # make some pooling layers
    meanpool = AGNPool("mean", 50, 10, 0.1)
    maxpool = AGNPool("max", 50, 10, 0.1)

    # start with the easy stuff
    @test meanpool(ag) == ones(Float32, 10, 1)
    @test maxpool(ag) == ones(Float32, 10, 1)

    # one level up
    feat[:,2] .= 0.0
    add_features!(ag, feat, dummyfzn)
    # they're still the same here because maxpool takes max along features but averages across nodes right now
    @test all(isapprox.(meanpool(ag), 2/3))
    @test all(isapprox.(maxpool(ag), 2/3))

    # and now make them different
    feat[1:5:50,2] .= 4.0
    add_features!(ag, feat, dummyfzn)
    @test all(isapprox.(meanpool(ag), 14/15))
    @test all(maxpool(ag) .== 2.0)
end
