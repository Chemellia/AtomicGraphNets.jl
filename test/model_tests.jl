using Test
using AtomicGraphNets
using SimpleWeightedGraphs
using GeometricFlux

@testset "Xie_model" begin
    # build a model with deterministic initialization of weights
    in_fea_len = 40
    conv_fea_len = 20
    pool_fea_len = 10
    model = Xie_model(in_fea_len, atom_conv_feature_length=conv_fea_len, pooled_feature_length=pool_fea_len, num_hidden_layers=2, initW=ones)

    # check that everything is the size it should be
    @test length(model)==5
    @test size(model[1].convweight) == size(model[1].selfweight) == (conv_fea_len, in_fea_len)
    @test size(model[2].convweight) == size(model[2].selfweight) == (conv_fea_len, conv_fea_len)
    @test size(model[4].W) == (pool_fea_len, pool_fea_len)
    @test size(model[5].W) == (1, pool_fea_len)

    # check that it evaluates to the right things, basically
    input = FeaturedGraph(SimpleWeightedGraph([0 1; 1 0]), ones(40, 2))
    @test isapprox(feature(model[1:2](input)),zeros(20,2),atol=1e-10)
    @test isapprox(model(input)[1], 6.9314718)
end