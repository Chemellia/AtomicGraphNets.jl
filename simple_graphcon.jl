#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#

#using PyCall
#using GeometricFlux
#using SimpleWeightedGraphs, MetaGraph
using GraphPlot, Colors
using CSV
include("graph_functions.jl")
include("featurize.jl")
include("layers.jl")

# TODO: think more about data structures
# need to maintain association between node indices and elements,
# among other things...
# maybe do a MetaGraph to store that stuff (and build featurization)
# then cast to SWG for actual learning...

# define some high-level options
num_conv = 3
#pool_func =
prop = "formation_energy_per_atom"
datadir = "../MP_data/"
id = "task_id"

# atom featurization, pretty arbitrary choices for now
features = ["group", "row", "block", "atomic_mass", "atomic_radius", "X"]
num_bins = [18, 6, 4, 20, 10, 10]
logspaced = [false, false, false, true, true, false]
atom_feature_vecs = make_feature_vectors(features, num_bins, logspaced)

# dataset...first, read in outputs
info = CSV.read(string(datadir,prop,".csv"))
y = info[!, Symbol(prop)]

# next, make graphs
graphs_elements = []
for r in eachrow(info[1:100,:])
    cifpath = string(datadir,prop,"_cifs/",r[Symbol(id)],".cif")
    append!(graphs_elements, [build_graph(cifpath)])
    # TODO: there is probably a smarter way to store this information...
end


# build the network
# ...

# train
# ...
