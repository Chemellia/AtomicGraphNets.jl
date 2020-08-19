using Test

include("../src/layers.jl")

@testset "CGCNConv" begin
    # create simple line graph, populate it with feature of all ones
    adjmat = Float32.([0 1 0; 1 0 1; 0 1 0])
    fg = FeaturedGraph(SimpleWeightedGraph(adjmat), ones(4,3))
    # create a conv layer, initialize weights with ones
    l = CGCNConv(4=>4, initW=ones)

    # test that two function signatures give same result
    @test feature(l(fg)) == feature(l(graph(fg).weights, feature(fg)))

    # test output looks as expected
    output_fea = feature(l(fg))
    @test output_fea[:,1]==output_fea[:,3]
    @test isapprox(output_fea[:,1].+output_fea[:,3], .-output_fea[:,2])

    # and now for a loop
    adjmat = Float32.([0 1 1; 1 0 1; 1 1 0])
    fg = FeaturedGraph(SimpleWeightedGraph(adjmat), ones(4,3))
    l = CGCNConv(4=>4, initW=ones)

    @test feature(l(fg)) == feature(l(graph(fg).weights, feature(fg)))
    @test CrystalGraphConvNets.reg_norm(softplus(4.0) .* ones(4,3)) == feature(l(fg))
end

@testset "pooling" begin
    # keep our little line graph, but give it more features
    adjmat = Float32.([0 1 0; 1 0 1; 0 1 0])
    feat = ones(50,3)
    fg = FeaturedGraph(SimpleWeightedGraph(adjmat), feat)

    # make some pooling layers
    meanpool = CGCNMeanPool(10, 0.1)
    maxpool = CGCNMaxPool(10, 0.1)

    # start with the easy stuff
    @test meanpool(fg) == ones(10,1)
    @test maxpool(fg) == ones(10,1)

    # one level up
    feat[:,2] .= 0.0
    fg = FeaturedGraph(SimpleWeightedGraph(adjmat), feat)
    # they're still the same here because maxpool takes max along features but averages across nodes right now
    @test all(meanpool(fg) .== 2/3)
    @test all(maxpool(fg) .== 2/3)

    # and now make them different
    feat[1:5:50,2] .= 4.0
    fg = FeaturedGraph(SimpleWeightedGraph(adjmat), feat)
    @test all(isapprox.(meanpool(fg), 14/15))
    @test all(maxpool(fg) .== 2.0)
end