#=
Code that pulls atomic data from online databases (currently just Materials Project) and saves to a DataFrame. You'll need to have `pymatgen` installed and accessible to `PyCall` (i.e. in your Julia Conda.jl environment) for this to work.
=#
using PyCall
using PeriodicTable
using DataFrames
using CSV

# get output path by walking relative paths, then switch back to current location
current_dir = pwd()
this_dir = @__DIR__
cd(this_dir)
cd("../data")
output_folder = pwd()
cd(current_dir)
output_path = joinpath(output_folder, "mp_atomic_data.csv")

global default_nbins = 10

# we can probably skip radioactive stuff for now...
max_atno = 83

all_elements = [e.symbol for e in elements[1:max_atno]]

# leave off noble gases too
nums_to_skip = [2, 10, 18, 36, 54]

# NEXT: make separate functions to build dataframes from qmpy or pymatgen and then save it to file, commit file to repo and have the default be just to read from the file

avail_features = ["Z", "group", "row", "block", "atomic_mass", "atomic_radius", "van_der_waals_radius", "X"]
global categorical_features = ["group", "row", "block"]
global categorical_feature_vals = Dict("group"=>1:18, "row"=>1:8, "block"=>["s", "p", "d", "f"])

# compile data... (and skip noble gases)
pt = pyimport("pymatgen.core.periodic_table")
atom_data = [pt.Element(all_elements[i]) for i in 1:max_atno if !(i in nums_to_skip)]
all_elements = [all_elements[i] for i in 1:max_atno if !(i in nums_to_skip)]

py"""
def isnone(x):
    return x is None
"""

# reformat into a DataFrame for niceness
global atom_data_df = DataFrame(sym = all_elements)
for feature in avail_features
    if feature=="block"
        feature_vals = ["a" for i in 1:length(all_elements)]
    elseif feature in ["Z", "group", "row"]
        feature_vals = missings(Int32, length(all_elements))
    else
        feature_vals = missings(Float32, length(all_elements))
    end
    for i in 1:length(all_elements)
        feature_val = getproperty(atom_data[i], feature)
        if !py"isnone"(feature_val)
            feature_vals[i] = feature_val
        end
    end
    global atom_data_df[!, Symbol(feature)] = feature_vals
end

# relabel rows for lanthanoids (should be 6 not 8)
#=
rows = atom_data_df.row
for i in 1:length(rows)
    if rows[i]==8.0
        rows[i] = 6.0
    end
end
atom_data_df.row = rows
=#

CSV.write(output_path, atom_data_df)
