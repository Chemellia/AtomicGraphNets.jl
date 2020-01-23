#=
Featurizing atomic data...

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
=#
using PyCall
using PeriodicTable
using DataFrames
using Flux:onehot

# we can probably skip radioactive stuff for now...
max_atno = 83
all_elements = [e.symbol for e in elements[1:max_atno]]

# leave off noble gases too
nums_to_skip = [2, 10, 18, 36, 54]

avail_features = ["group", "row", "block", "atomic_mass", "atomic_radius", "van_der_waals_radius", "X"]
categorical_features = ["group", "row", "block"]
categorical_feature_vals = Dict("group"=>1:18, "row"=>1:6, "block"=>["s", "p", "d", "f"])

# compile data... (and skip noble gases)
pt = pyimport("pymatgen.core.periodic_table")
atom_data = Dict(i => pt.Element(all_elements[i]) for i in 1:max_atno)

py"""
def isnone(x):
    return x is None
"""

# reformat into a DataFrame for niceness
atom_data_df = DataFrame(sym = all_elements)
for feature in avail_features
    if feature=="block"
        feature_vals = ["a" for i in 1:max_atno]
    else
        feature_vals = missings(Float64, max_atno)
    end
    for i in keys(atom_data)
        feature_val = getproperty(atom_data[i], feature)
        if !py"isnone"(feature_val)
            feature_vals[i] = feature_val
        end
    end
    atom_data_df[!, Symbol(feature)] = feature_vals
end

# drop rows we don't need...
atom_data_df = atom_data_df[[i for i in 1:max_atno if !(i in nums_to_skip)], :]
all_elements = [all_elements[i] for i in 1:max_atno if !(i in nums_to_skip)]

# relabel rows for lanthanoids (should be 6 not 8)
rows = atom_data_df.row
for i in 1:size(rows,1)
    if rows[i]==8.0
        rows[i] = 6.0
    end
end
atom_data_df.row = rows

# return bin edges for a given feature given the corresponding column in the DataFrame
function make_bins(df_col; n_bins=10, logspace=false)
    min_val = minimum(skipmissing(df_col))
    max_val = maximum(skipmissing(df_col))
    if logspace
        bin_edges = 10 .^range(log10(min_val), log10(max_val), length=n_bins+1)
    else
        bin_edges = range(min_val, max_val, length=n_bins+1)
    end

    return bin_edges
end

#=
Figure out which bin val sits in, return vector of 0's and a single 1 in the right bin.

If it's on an edge it should end up in the lower bin (by my arbitrary choice, plus it probably will only happen for the absolute min and max values).
=#
function onehot_bins(val, bin_edges)
    onehot_vec = [false for i in 1:size(bin_edges,1)-1]
    for i in 1:size(bin_edges, 1)-1
        if (val >= bin_edges[i]) & (val <= bin_edges[i+1])
            onehot_vec[i] = 1
        end
    end
    return onehot_vec
end

#=
Make custom feature vectors, using specified features and numbers of bins. Note that bin numbers will be ignored for categorical features (block, group, and row), but features and nbins vectors should still be the same length (there's probably a more elegant way to handle that).

Optionally, feed in vector of booleans with trues at the index of any feature whose bins should be log spaced.

Returns a dictionary from element symbol => one-hot style feature vector, concatenated in order of feature list.
=#
function make_feature_vectors(features, nbins; atom_data_df=atom_data_df, logspaced=false)
    # figure out spacing
    if logspaced==false # default behavior
        logspaced_vec = [false for i in 1:size(features,1)]
    elseif logspaced==true
        logspaced_vec = [true for i in 1:size(features,1)]
    elseif size(logspaced, 1) == size(features, 1) # specified properly
        logspaced_vec = logspaced
    elseif size(logspaced, 1) < size(features, 1)
        println("logspaced vector too short. Padding end with falses.")
        logspaced_vec = hcat(logspaced, [false for i in 1:size(features,1)-size(logspaced,1)])
    elseif size(logspaced, 1) > size(features, 1)
        println("logspaced vector too long. Cutting off at appropriate length.")
        logspaced_vec = logspaced[1:size(features,1)]
    end

    # make dict from feature name to bin edge list for continuous-valued features
    features_bins = Dict(features[i] => make_bins(atom_data_df[:, Symbol(features[i])]; n_bins=Int64(nbins[i]), logspace=logspaced_vec[i]) for i in 1:size(features,1) if !(features[i] in categorical_features))

    sym_featurevec = Dict()
    # for categorical features, can use onehot from Flux
    for i in 1:size(atom_data_df,1)
        el = atom_data_df.sym[i]
        featurevec = []
        # make onehot vector for each feature
        for feature in features
            feature_val = atom_data_df[i, Symbol(feature)]
            if feature in categorical_features
                subvec = onehot(feature_val, categorical_feature_vals[feature])
            else
                subvec = onehot_bins(feature_val, features_bins[feature])
            end
            append!(featurevec, subvec)
        end
        sym_featurevec[el] = featurevec
    end
    return sym_featurevec
end

# helper to subdivide vec according to nbins
function chunk_vec(vec, nbins)
    chunks = fill(Bool[], size(nbins, 1))
    if !(size(vec,1)==sum(nbins))
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

#=
Function for debugging. Just checks that each subvector in a featurization is a valid one-hot encoding (one true and otherwise all falses).
=#
function vec_valid(vec, nbins)
    result = true
    # are there the right number of total bins?
    chunks = chunk_vec(vec, nbins)
    if chunks == fill(Bool[], size(nbins, 1))
        result = false
    else
        # does each subvector have exactly one true value?
        for i in 1:size(nbins, 1)
            subvec = chunks[i]
            if !(sum(subvec)==1)
                println("Subvector ", i, " is invalid.")
                result = false
            end
        end
    end
    return result
end

#=
Function to invert the binning process. Useful to check that it's working properly, or just to inspect properties once they've been encoded.

Need to feed in a feature vector as well as the lists of features and bin numbers that were used to encode it, and the dataframe containing the atomic data.
=#
function decode_feature_vector(vec, features, nbins; atom_data_df=atom_data_df)
    # First, check that the featurization is valid
    if !(vec_valid(vec))
        println("Vector is invalid!")
    else
        chunks = chunk_vec(vec, nbins)
        # ...
    end
end

# next: rigorous check of featurization for different features, spacings, bin numbers, etc.
