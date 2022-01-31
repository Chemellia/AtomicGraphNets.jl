import ChemistryFeaturization: encode, encodable_elements, decode, output_shape, features
using ChemistryFeaturization.Data
using ChemistryFeaturization.ElementFeature

using DataFrames

const default_nbins = 10

"""
    GraphNodeFeaturization(atom_features, codecs)
    GraphNodeFeaturization(feature_names, lookup_table; nbins, logspaced, categorical)

A featurization for AtomGraph objects that encodes features associated with each node. Contains a collection of `AbstractAtomFeatureDescriptor` objects, and can be initialized by passing those, or by passing parameters for constructing them.

## Required Arguments
- `feature_names::Vector{String}`: Names of each feature
- `lookup_table::DataFrame` (optional): source of data for features, if not included in built-in `element_data_df`

## Keyword Arguments
- `nbins::Union{Vector{Integer},Integer}`: Number of bins to use for one-hot encoding of continuous-valued features. Will be ignored for categorical features.
- `logspaced::Union{Vector{Bool},Bool}`: Whether to logarithmically space the bins
- `categorical::Union{Vector{Bool},Bool}`: Whether each feature is categorical or continous-valued.
"""
struct GraphNodeFeaturization <: AbstractFeaturization
    features::Vector{<:AbstractAtomFeatureDescriptor}
    codecs::Vector{<:AbstractCodec}
end

GraphNodeFeaturization(features::Vector{<:AbstractAtomFeatureDescriptor}) =
    GraphNodeFeaturization(features, default_codec.(features))

# NB: this constructor only works if every feature is an ElementFeature
function GraphNodeFeaturization(
    feature_names::Vector{String},
    lookup_table::Union{DataFrame,Nothing} = nothing;
    nbins::Union{Vector{<:Integer},Integer,Nothing} = nothing,
    logspaced::Union{Vector{Bool},Bool,Nothing} = nothing,
    categorical::Union{Vector{Bool},Bool,Nothing} = nothing,
)
    num_features = length(feature_names)
    local lookup_table_here, logspaced_here, categorical_here, nbins_here
    if isnothing(lookup_table)
        lookup_table_here = element_data_df
    else
        # need to merge them in case some data is in one place and some the other
        lookup_table_here = outerjoin(element_data_df, lookup_table, on = :Symbol)
    end

    if isnothing(logspaced)
        logspaced_here = map(fn -> default_log(fn, lookup_table_here), feature_names)
    else
        logspaced_here = get_param_vec(logspaced, num_features)
    end

    if isnothing(categorical)
        categorical_here =
            map(fn -> default_categorical(fn, lookup_table_here), feature_names)
    else
        categorical_here = get_param_vec(categorical, num_features)
    end

    if isnothing(nbins)
        nbins_here = [default_nbins for i = 1:num_features]
    else
        nbins_here = get_param_vec(nbins, num_features, pad_val = default_nbins)
    end

    bins = map(zip(feature_names, nbins_here, logspaced_here, categorical_here)) do args
        get_bins(
            args[1],
            lookup_table_here,
            nbins = args[2],
            logspaced = args[3],
            categorical = args[4],
        )
    end

    codecs = OneHotOneCold.(categorical_here, bins)
    efs = ElementFeatureDescriptor.(feature_names, Ref(lookup_table_here))

    GraphNodeFeaturization(efs, codecs)
end

# pretty printing, short
function Base.show(io::IO, fzn::GraphNodeFeaturization)
    st = "GraphNodeFeaturization encoding $(length(fzn.features)) features"
    print(io, st)
end

# pretty printing, long
function Base.show(io::IO, ::MIME"text/plain", fzn::GraphNodeFeaturization)
    st = "GraphNodeFeaturization encoding $(length(fzn.features)) features:\n"
    for feature in fzn.features
        st = string(st, "\t", feature, "\n")
    end
    st = st[1:end-1]
    print(io, st)
end

output_shape(fzn::GraphNodeFeaturization) = sum(output_shape.(fzn.codecs))

features(fzn::GraphNodeFeaturization) = fzn.features

"""
    chunk_vec(vec, nbins)

Helper function that divides up an already-constructed feature vector into "chunks" (one for each feature) of lengths specified by the vector nbins.

Sum of nbins should be equal to the length of vec.

# Examples
```jldoctest
julia> chunk_vec([1,0,0,1,0], [3,2])
2-element Array{Array{Bool,1},1}:
 [1, 0, 0]
 [1, 0]
 ```
"""
function chunk_vec(vec::Vector{<:Real}, nbins::Vector{<:Integer})
    chunks = fill(Bool[], size(nbins, 1))
    @assert length(vec) == sum(nbins) "Total number of bins doesn't match length of feature vector."
    for i = 1:size(nbins, 1)
        if i == 1
            start_ind = 1
        else
            start_ind = sum(nbins[1:i-1]) + 1
        end
        chunks[i] = vec[start_ind:start_ind+nbins[i]-1]
    end
    return chunks
end

function encode(ag::AtomGraph, fzn::GraphNodeFeaturization)
    encoded = reduce(hcat, map(x -> encode(ag, x[1], x[2]), zip(fzn.features, fzn.codecs)))
    return encoded
end

function decode(encoded::Matrix{<:Real}, fzn::GraphNodeFeaturization)
    num_atoms = size(encoded, 1)
    nbins = [output_shape(c) for c in fzn.codecs]
    local decoded = Dict{Integer,Dict{String,Any}}()
    for i = 1:num_atoms
        #println("atom $i")
        chunks = chunk_vec(encoded[i, :], nbins)
        decoded[i] = Dict{String,Any}()
        for (chunk, f, c) in zip(chunks, fzn.features, fzn.codecs)
            #println("    $(f.name): $(decode(f, chunk))")
            decoded[i][f.name] = decode(chunk, c)
        end
    end
    return decoded
end
