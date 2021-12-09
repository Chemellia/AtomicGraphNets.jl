using Test
using AtomicGraphNets
using ChemistryFeaturization
using Flux

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
        initW = (d...) -> fill(0.1, d...),
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
        @test model(input)[1] ≈ 0.693 atol = 1e-3
    end

    @testset "backward pass" begin
        loss(x, y) = Flux.Losses.mse(model(x), y)
        data = zip([input], [1.0]) # high-quality data...
        opt = Descent(0.1) # should be deterministic
        Flux.train!(loss, params(model), data, opt)
        @test model(input)[1] ≈ 1.051 atol = 1e-3
    end
end

@testset "SGCNN" begin
    # could probably do with some more detailed tests here but better something than nothing for now
    model = build_SGCNN(40, initW = (d...) -> fill(0.1, d...))

    @testset "initialization" begin
        @test length(model) == 5
        @test model[1].connection == vcat
        @test length.(model[1].layers) == (3, 3)
    end

    @testset "forward pass" begin
        @test model((input, input))[1] ≈ 45.331 atol = 1e-3
    end

    @testset "backward pass" begin
        loss(x, y) = Flux.Losses.mse(model(x), y)
        data = zip([(input, input)], [1.0]) # high-quality data...
        opt = Descent(0.1) # should be deterministic
        Flux.train!(loss, params(model), data, opt)
        @test model((input, input))[1] ≈ -75.985 atol = 1e-3
    end
end

@testset "DEQ" begin
    # build a model with deterministic initialization of weights
    in_fea_len = 40
    pool_fea_len = 10
    model = build_CGCNN_DEQ(
        in_fea_len,
        pooled_feature_length = pool_fea_len,
        num_hidden_layers = 2,
        initW = (d...) -> fill(0.1, d...),
    )

    @testset "initialization" begin
        @test length(model) == 4
        @test size(model[1].conv.convweight) ==
              size(model[1].conv.selfweight) ==
              (in_fea_len, in_fea_len)
        @test size(model[3].weight) == (pool_fea_len, pool_fea_len)
        @test size(model[4].weight) == (1, pool_fea_len)
    end

    @testset "forward pass" begin
        @test model(input)[1] ≈ 0.693147 atol = 1e-6
    end

    @testset "backward pass" begin
        loss(x, y) = Flux.Losses.mse(model(x), y)
        data = zip([input], [1.0]) # high-quality data...
        opt = Descent(0.1) # should be deterministic
        Flux.train!(loss, params(model), data, opt)
        @test model(input)[1] ≈ 1.051563 atol = 1e-6
    end
end
