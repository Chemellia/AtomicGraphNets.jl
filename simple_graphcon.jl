#using PyCall
using GeometricFlux
using SimpleWeightedGraphs
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

# dataset...first, read in outputs
info = CSV.read(string(datadir,prop,".csv"))
y = info[!, Symbol(prop)]

# next, make graphs
input_graphs = []
for r in eachrow(info)
    cifpath = string(datadir,prop,"_cifs/",r[Symbol(id)],".cif")
    append!(input_graphs, build_graph(cifpath))

# featurization
# ...

# build the network
# ...

# train
# ...
