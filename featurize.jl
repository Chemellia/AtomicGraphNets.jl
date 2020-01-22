#=
Featurizing atomic data...

Features to include (easy, all in pymatgen with these names)
* group
* row
* block (s, p, d, f)
* atomic_mass
* atomic_radius
* van_der_waals_radius
* X (Pauling electronegativity)

Others to look up/figure out
* electron affinity
* (first) ionization energy
* valence electrons
* atomic and vdw radii both have missing values

Things I'd like, but may be harder
* oxidation states (one-hot across all possibilities of whether it exists or not, but do we include only "common" ones or absolutely all?)
* later ionization energies (small elements don't have them)
* energy levels (how to handle varying numbers of electrons, etc. – could just count backwards from vacuum?) [atomic_orbitals in pmg]
=#
using PyCall
using PeriodicTable
using DataFrames

# the first 103 ought to be enough for anyone...
all_elements = [e.symbol for e in elements[1:103]]
avail_features = ["group", "row", "block", "atomic_mass", "atomic_radius", "van_der_waals_radius", "X"]

# compile data...
pt = pyimport("pymatgen.core.periodic_table")
atom_data = Dict(i => pt.Element(all_elements[i]) for i in 1:103)

py"""
def isnone(x):
    return x is None
"""

# reformat into a DataFrame for niceness
atom_data_df = DataFrame(sym = all_elements)
for feature in avail_features
    if feature=="block"
        feature_vals = ["a" for i in 1:103]
    else
        feature_vals = missings(Float64, 103)
    end
    for i in 1:103
        feature_val = getproperty(atom_data[i], feature)
        if !py"isnone"(feature_val)
            feature_vals[i] = feature_val
        end
    end
    atom_data_df[!, Symbol(feature)] = feature_vals
end

#=
Next:
* figure out how to handle NaNs/missing values when calculating ranges for binning
* also how to deal with them in featurizing...
=#

function make_bins(min_val, max_val; n_bins=10, scaling='linear')
    # do things

    # return vector of n_bins + 1 values for bin edges
end

function onehot_bins(val, bin_edges)
    # figure out which bin val sits in
    # return vector of 0's and a single 1 in the right bin
end

function make_feature_vectors(features_bins; element_list=all_elements)
    # input should be a dict of feature_name => num_bins
    # num_bins will be ignored for block, period, and group

    for k in keys(features_bins)
        # download data from pymatgen (use as_dict() to pass in strings)
        # make and store bins
    end

    # iterate over each element, download and bin data

    # return the concatenated vector, in order of dict keys
end
