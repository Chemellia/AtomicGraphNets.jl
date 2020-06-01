#=
Featurizing atomic data...

TODO: possibly add ability to include structure-specific features e.g. coordination,
then would need vector specific to each point in the structure rather than just each
unique element

Other features to look up/figure out:
* electron affinity
* (first) ionization energy
* valence electrons
* vdw radius has a lot of (34) missing values
(get them with print(atom_data_df[findall(x -> ismissing(x), atom_data_df.van_der_waals_radius), :]))

Things I'd like, but may be harder:
* oxidation states (one-hot across all possibilities of whether it exists or not, but do we include only "common" ones or absolutely all?)
* later ionization energies (small elements don't have them)
* energy levels (how to handle varying numbers of electrons, etc. – could just count backwards from vacuum?) [atomic_orbitals in pmg]

Other decisions to make:
* Should ranges of values (over which to bin) be fixed even if we're only using a subset of elements that don't contain the full range? For example, electronegativity ranges from 0.79 to 3.98 but if a subset of values only goes from 0.98 to 2.0 should the bins go over the whole range for consistency across different analyses, even though some bins would never be occupied for that subset?
* Is there a more elegant/abstracted way to distinguish between categorical and continuous features?

NOTE: if we experiment with other featurizations that don't just contain 0's and 1's, normalized vs. non-normalized graph Laplacians will become important!
=#
#using PyCall
#using PeriodicTable
using CSV
using DataFrames
using Flux:onehot, onecold

# copied from get_atomic_data currently, bad form
avail_features = ["Z", "group", "row", "block", "atomic_mass", "atomic_radius", "van_der_waals_radius", "X"]
global categorical_features = ["group", "row", "block"]
global categorical_feature_vals = Dict("group"=>1:18, "row"=>1:8, "block"=>["s", "p", "d", "f"])

# get data path and read in
current_dir = pwd()
this_dir = @__DIR__
cd(this_dir)
cd("../data")
data_folder = pwd()
cd(current_dir)
data_path = joinpath(data_folder, "mp_atomic_data.csv")
global atom_data_df = CSV.read(data_path)

# compile min and max values of each feature...
global fea_minmax = Dict()
for feature in avail_features
    if !(feature in categorical_features)
        minval = minimum(skipmissing(atom_data_df[:, Symbol(feature)]))
        maxval = maximum(skipmissing(atom_data_df[:, Symbol(feature)]))
        fea_minmax[feature] = (minval, maxval)
    end
end

# TODO: this
#function get_default_nbins()
#end

"Get bins for a given feature, intelligently handling categorical vs. continuous feature values. In the former case, returns the categories. In the later, returns bin edges."
function get_bins(feature; nbins=default_nbins, logspaced=false)
    if feature in categorical_features
        bins = categorical_feature_vals[feature]
    else
        min_val = fea_minmax[feature][1]
        max_val = fea_minmax[feature][2]
        if logspaced
            bins = 10 .^range(log10(min_val), log10(max_val), length=nbins+1)
        else
            bins = range(min_val, max_val, length=nbins+1)
        end
    end
    return bins
end


"Find which bin index a value sits in, intelligently handling both categorical and continous feature values."
function which_bin(feature, val, bins=get_bins(feature))
    if feature in categorical_features
        bin_index = findfirst(isequal(val), bins)
    else
        bin_index = searchsorted(bins, val).stop
        if bin_index == length(bins) # got the max value
            bin_index = bin_index-1
        end
    end
    return bin_index
end

"""
    onehot_bins(feature, val)
    onehot_bins(feature, val, bins)

Create onehot style vector, handling both categorical and continuous features.

# Arguments
- `feature::String`: feature being encoded
- `val`: (Float or String) value of feature to encode
- `bins::Array`: categorical values or bin edges if continouus feature

# Examples
```jldoctest
julia> onehot_bins("cont_feature", 3, [0,2,4,6])
3-element Array{Bool,1}:
 0
 1
 0

julia> onehot_bins("cat_feature", 3, [1,2,3])
3-element Array{Bool,1}:
 0
 0
 1
```

See also: [onecold_bins](@ref), [onehot](@ref Flux.onehot)

"""
function onehot_bins(feature, val, bins=get_bins(feature))
    if feature in categorical_features
        len = length(bins)
    else
        len = length(bins)-1
    end
    onehot_vec = [0.0 for i in 1:len]
    onehot_vec[which_bin(feature, val, bins)] = 1.0
    return onehot_vec
end

"""
    onecold_bins(feature, vec, bins)

Inverse function to onehot_bins, decodes a vector corresponding to a given feature, given the bins that were used to encode it.

# Arguments
- `feature::String`: feature being encoded
- `vec::Array`: vector (such as produced by [onehot_bins](@ref))
- `bins::Array`: categorical values or bin edges if continouus feature

# Examples
```jldoctest
julia> onecold_bins("cont_feature", [0,1,0], [0,2,4,6])
(2,4)

julia> onecold_bins("cat_feature", [0,0,1], [1,2,3])
3
```

See also: [onehot_bins](@ref), [onecold](@ref Flux.onecold)

"""
function onecold_bins(feature, vec, bins)
    if feature in categorical_features
        # return value
        decoded = onecold(vec, bins)
    else
        # return range of values
        decoded = (onecold(vec, bins[1:end-1]), onecold(vec, bins[2:end]))
    end
    return decoded
end


"Little helper function to check that the logspace vector/boolean is appropriate and convert it to a vector as needed."
function get_logspaced_vec(vec, num_features)
    if vec==false # default behavior
        logspaced_vec = [false for i in 1:num_features]
    elseif vec==true
        logspaced_vec = [true for i in 1:num_features]
    elseif length(vec) == num_features # specified properly
        logspaced_vec = vec
    elseif length(vec) < num_features
        println("logspaced vector too short. Padding end with falses.")
        logspaced_vec = hcat(vec, [false for i in 1:num_features-size(vec,1)])
    elseif size(vec, 1) > num_features
        println("logspaced vector too long. Cutting off at appropriate length.")
        logspaced_vec = vec[1:num_features]
    end
    return logspaced_vec
end

"""
    make_feature_vectors(features)
    make_feature_vectors(features, nbins)
    make_feature_vectors(features, nbins, logspaced)

Make custom feature vectors, using specified features and numbers of bins. Note that bin numbers will be ignored for categorical features (block, group, and row), but features and nbins vectors should still be the same length (there's probably a more elegant way to handle that).

Optionally, feed in vector of booleans with trues at the index of any (continous valued) feature whose bins should be log spaced.

# Arguments
- `features::Array{String,1}`: list of features to be encoded
- `nbins::Array{Integer,1}`: number of bins for each feature (in same order)
- `logspaced::Array{Bool,1}=false`: whether or not to logarithmically space each feature

Returns a dictionary from element symbol => one-hot style feature vector, concatenated in order of feature list.
"""
function make_feature_vectors(features, nbins=default_nbins*ones(Int64, size(features,1)), logspaced=false)
    num_features = size(features,1)

    # figure out spacing for each feature
    logspaced_vec = get_logspaced_vec(logspaced, num_features)

    # make dict from feature name to bins
    features_bins = Dict(features[i] => get_bins(features[i]; nbins=Int64(nbins[i]), logspaced=logspaced_vec[i]) for i in 1:num_features)

    # dict from feature name to number of bins for that feature
    features_nbins = Dict(zip(features, nbins))

    # dict from element symbol to feature vec of that element
    # (if we do any structure-specific features later, e.g. coordination or something,
    # this will have to iterate over every atom in the structure instead...)
    # (but possibly would just want to append those to the end anyway...)
    sym_featurevec = Dict{String, Array{Float32,1}}()
    for i in 1:size(atom_data_df,1)
        el = atom_data_df.sym[i]
        featurevec = []
        # make onehot vector for each feature
        for feature in features
            feature_val = atom_data_df[i, Symbol(feature)]
            subvec = onehot_bins(feature, feature_val, get_bins(feature; nbins=features_nbins[feature]))
            append!(featurevec, subvec)
        end
        sym_featurevec[el] = featurevec # need transpose because of how graphcon works
    end
    return sym_featurevec
end

"""
    chunk_vec(vec, nbins)

Divide up an already-constructed feature vector into "chunks" (presumably one for each feature) of lengths specified by the vector nbins.

Sum of nbins should be equal to the length of vec.

# Examples
```jldoctest
julia> chunk_vec([1,0,0,1,0], [3,2])
2-element Array{Array{Bool,1},1}:
 [1, 0, 0]
 [1, 0]
 ```
"""
function chunk_vec(vec, nbins)
    chunks = fill(Bool[], size(nbins, 1))
    if !(length(vec)==sum(nbins))
        println("Total number of bins doesn't match length of feature vector.")
        return chunks
    else
        for i in 1:size(nbins,1)
            if i==1
                start_ind = 1
            else
                start_ind = sum(nbins[1:i-1])+1
            end
            chunks[i] = vec[start_ind:start_ind+nbins[i]-1]
        end
    end
    return chunks
end

"""
Check that each subvector in a featurization is a valid one-hot encoding (one true and otherwise all falses).
"""
function vec_valid(vec, nbins)
    result = true
    # are there the right number of total bins?
    chunks = chunk_vec(vec, nbins)
    if chunks == fill(Bool[], length(nbins))
        result = false
    else
        # does each subvector have exactly one true value?
        for i in 1:length(nbins)
            subvec = chunks[i]
            if !(sum(subvec)==1)
                println("Subvector ", i, " is invalid.")
                result = false
            end
        end
    end
    return result
end

"""
Function to invert the binning process. Useful to check that it's working properly, or just to inspect properties once they've been encoded.

Need to feed in a feature vector as well as the lists of features and bin numbers that were used to encode it.
"""
function decode_feature_vector(vec, features, nbins, logspaced=false)
    # First, check that the featurization is valid
    if !(vec_valid(vec, nbins))
        println("Vector is invalid!")
    else
        chunks = chunk_vec(vec, nbins)
        # make dict from features to corresponding chunks
        fea_chunks = Dict(zip(features, chunks))

        # and one from feature to bin bounds for this vector
        num_features = length(features)
        logspaced_vec = get_logspaced_vec(logspaced, num_features)
        fea_bins = Dict(features[i]=>get_bins(features[i]; nbins=nbins[i], logspaced=logspaced_vec[i]) for i in 1:num_features)

        return Dict(feature=>onecold_bins(feature, fea_chunks[feature], fea_bins[feature]) for feature in features)
    end
end
