using ChemistryFeaturization, AtomGraphs
using ChemistryFeaturization.ElementFeature
using CSV, DataFrames

# custom lookup table...
test_df = CSV.read(abspath(@__DIR__, "test_data", "lookup_table.csv"), DataFrame)

fnames = ["X", "Block", "Atomic mass"]
fs = ElementFeatureDescriptor.(fnames)

triangle_C = AtomGraph(Float32.([0 1 1; 1 0 1; 1 1 0]), ["C", "C", "C"])

@testset "Encode/Featurize" begin
    using ChemistryFeaturization.ElementFeature

    # make sure both constructors give the same results
    fzn1 = GraphNodeFeaturization(fs)
    fzn2 = GraphNodeFeaturization(fnames)
    fzn3 = GraphNodeFeaturization(fnames, nbins=[6,4,10])

    @test output_shape(fzn1) == 24 # 10 (default length for X) + 4 (default for block) + 10 (default for atomic mass)
    @test output_shape(fzn2) == 24
    @test output_shape(fzn3) == 20

    encoded_1, encoded_2 = encode.([triangle_C, triangle_C], Ref(fzn1)) # encode can be broadcasted
    @test encoded_1 == encoded_2

    encoded_1, encoded_2 = encode.([triangle_C, triangle_C], [fzn1, fzn2]) # encode can be broadcasted
    @test encoded_1 == encoded_2

    featurized_1, featurized_2 = featurize.([triangle_C, triangle_C], [fzn1, fzn2])
    @test featurized_1.encoded_features == featurized_2.encoded_features
end

@testset "Decode" begin
    fzn1 = GraphNodeFeaturization(fs)
    fa = featurize(triangle_C, fzn1)
    decoded = decode(fa)

    @test all(map(i -> decoded[i]["Block"] == "p", 1:3))
    @test all(map(i -> decoded[i]["X"][1] <= 2.55 <= decoded[i]["X"][2], 1:3))
end

# encodable_elements
@testset "Encodable Elements" begin
    fzn4 = GraphNodeFeaturization(ElementFeatureDescriptor.(["Boiling point", "6d"]))
    fzn5 = GraphNodeFeaturization(ElementFeatureDescriptor.(["7s", "Valence"]))
    @test encodable_elements(fzn4) == ["Ac", "Th", "U"]
    @test encodable_elements(fzn5) == ["Fr", "Ra", "Ac", "Th"]

    # Custom lookup_table
    feature_1 = ElementFeatureDescriptor("MeaningOfLife", test_df)   # zero-value case - `As` has a value = 0
    feature_2 = ElementFeatureDescriptor("neg_nums", test_df)
    feature_3 = ElementFeatureDescriptor("first_letter", test_df)
    feature_4 = ElementFeatureDescriptor("noarsenic", test_df) # missing value case - `As` has a missing value

    @test encodable_elements(GraphNodeFeaturization([feature_1, feature_2])) ==
            ["C", "As", "Tc"]
    @test encodable_elements(GraphNodeFeaturization([feature_1, feature_4])) ==
            ["C", "Tc"]
    @test encodable_elements(GraphNodeFeaturization([feature_2, feature_3])) ==
            ["C", "As", "Tc"]
    @test encodable_elements(GraphNodeFeaturization([feature_2, feature_4])) ==
            ["C", "Tc"]
end

# chunk_vec helper fcn
@testset "chunk_vec" begin
    vec = [1, 1, 0, 1, 0, 1, 0]
    @test_throws AssertionError chunk_vec(vec, [3, 3])
    @test chunk_vec(vec, [4, 1, 2]) == [[1, 1, 0, 1], [0], [1, 0]]
end
