using Test
using AtomicGraphNets
using ChemistryFeaturization
using BSON: @save

using Pkg

# define a test model input
dummyfzn = GraphNodeFeaturization(["X" for i = 1:40], nbins = 1) # cheeky way to get a matrix of all 1's
ag = AtomGraph(Float64.([0 1; 1 0]), ["H", "O"])
input = featurize(ag, dummyfzn)

@testset "CGCNN" begin
    # build a model with deterministic initialization of weights
    in_fea_len = 40
    conv_fea_len = 20
    pool_fea_len = 10
    model = build_CGCNN(
        in_fea_len,
        atom_conv_feature_length = conv_fea_len,
        pooled_feature_length = pool_fea_len,
        num_hidden_layers = 2,
        initW = ones,
    )

    @testset "initialization" begin
        @test length(model) == 5
        @test size(model[1].convweight) ==
            size(model[1].selfweight) ==
            (conv_fea_len, in_fea_len)
        @test size(model[2].convweight) ==
            size(model[2].selfweight) ==
            (conv_fea_len, conv_fea_len)
        @test size(model[4].weight) == (pool_fea_len, pool_fea_len)
        @test size(model[5].weight) == (1, pool_fea_len)
    end

    @testset "forward pass" begin
        lapl, output1 = model[1](input)
        int_mat =
            model[2].convweight * output1 * lapl +
            model[2].selfweight * output1 +
            hcat([model[2].bias for i = 1:size(output1, 2)]...)
        @test all(isapprox.(model[1:2](input)[2], zeros(Float64, 20, 2), atol = 2e-3))
        @test isapprox(model(input)[1], 6.93, atol = 1e-2)
    end

    # TODO: these
    # @testset "backward pass" begin

    # end
end

@testset "SGCNN" begin
    # could probably do with some more detailed tests here but better something than nothing for now
    model = build_SGCNN(40, initW=ones)

    @testset "initialization" begin
        @test length(model) == 5
        @test model[1].connection == vcat
        @test length.(model[1].layers) == (3, 3)
    end

    @testset "forward pass" begin
        @test isapprox(model((input, input))[1], 44361.42, atol=1e-3)
    end

    @testset "backward pass" begin

    end
end
