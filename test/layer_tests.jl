using Test
using ChemistryFeaturization
using SimpleWeightedGraphs

@testset "AGNConv" begin
    # create simple line graph, populate it with feature of all ones
    adjmat = Float64.([0 1 0; 1 0 1; 0 1 0])
    ag = AtomGraph(adjmat, ["C", "C", "C"])
    dummyfzn = GraphNodeFeaturization(["Block"])
    fa = featurize(ag, dummyfzn)

    # create a conv layer, initialize weights with ones
    l = AGNConv(4 => 4, initW = ones)

    # test output looks as expected
    output_fea = l(fa)[2]
    @test output_fea[1, :] == output_fea[3, :]
    @test isapprox(output_fea[1, :] .+ output_fea[3, :], .-output_fea[2, :])

    # and now for a loop
    adjmat = Float64.([0 1 1; 1 0 1; 1 1 0])
    ag = AtomGraph(adjmat, ["C", "C", "C"])
    fa = featurize(ag, dummyfzn)
    @test all(isapprox.(l(fa)[2], zero(Float64), atol = 1e-12))
end

@testset "AGNConvDEQ" begin
    # create simple line graph, populate it with feature of all ones
    adjmat = Float64.([0 1 0; 1 0 1; 0 1 0])
    ag = AtomGraph(adjmat, ["C", "C", "C"])
    dummyfzn = GraphNodeFeaturization(["Block"])
    fa = featurize(ag, dummyfzn)

    # create a conv layer, initialize weights with ones
    l = AGNConvDEQ(4, initW = ones, initb = zeros)

    # test output looks as expected
    output_fea = l(fa)[2]
    @test output_fea[1, :] == output_fea[3, :]
    @test isapprox(output_fea[1, :] .+ output_fea[3, :], .-output_fea[2, :])

    # and now for a loop
    adjmat = Float64.([0 1 1; 1 0 1; 1 1 0])
    ag = AtomGraph(adjmat, ["C", "C", "C"])
    fa = featurize(ag, dummyfzn)
    @test all(isapprox.(l(fa)[2], zero(Float64), atol = 1e-12))
end

@testset "pooling" begin
    # keep our little line graph, but give it more features
    adjmat = Float64.([0 1 0; 1 0 1; 0 1 0])
    feat = ones(Float64, 3, 50)

    # make some pooling layers
    meanpool = AGNPool("mean", 50, 10, 0.1)
    maxpool = AGNPool("max", 50, 10, 0.1)

    # start with the easy stuff
    # (also actually test that it evaluates on the FeaturizedAtoms too)
    @test meanpool(feat) == ones(Float64, 10, 1)
    @test maxpool(feat) == ones(Float64, 10, 1)

    # one level up
    feat[2, :] .= 0.0
    # they're still the same here because maxpool takes max along features but averages across nodes right now
    @test all(isapprox.(meanpool(feat), 2 / 3))
    @test all(isapprox.(maxpool(feat), 2 / 3))

    # and now make them different
    feat[2, 1:5:50] .= 4.0
    @test all(isapprox.(meanpool(feat), 14 / 15))
    @test all(maxpool(feat) .== 2.0)

    # make sure it complains when it should
    @test_throws AssertionError AGNPool("mean", 10, 20, 0.5)
end
