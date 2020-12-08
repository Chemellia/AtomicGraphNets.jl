using Test
using AtomicGraphNets
using ChemistryFeaturization
using SimpleWeightedGraphs

@testset "CGCNN" begin
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
    dummyfzn = [AtomFeat(:feat, [0]) for i in 1:40]
    input = AtomGraph(SimpleWeightedGraph{Int32,Float32}([0 1; 1 0]), ["H", "O"], ones(Float32, 40, 2), dummyfzn)
    output1 = model[1](input)
    #println(output1.features)
    int_mat = model[2].convweight * output1.features * output1.lapl + model[2].selfweight * output1.features + hcat([model[2].bias for i in 1:size(output1.features, 2)]...)
    #println(int_mat)
    #println(model[2].σ.(int_mat))
    #println(AtomicGraphNets.reg_norm(model[2].σ.(int_mat)))
    # TODO: figure out why the reg_norm step gives different results in REPL than in testing
    @test all(isapprox.(model[1:2](input).features, zeros(Float32, 20,2), atol=2e-3))
    @test isapprox(model(input)[1], 6.9, atol=3e-2)
end
