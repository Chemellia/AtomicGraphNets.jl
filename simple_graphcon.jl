#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#

using GraphPlot, Colors
using CSV
using Random
include("graph_functions.jl")
include("featurize.jl")
include("layers.jl")

# data-related options
num_pts = 100 # how many points to use? Up to 32530
train_frac = 0.2 # what fraction for training?
num_train = Int32(round(train_frac) * num_pts)
num_test = num_pts - num_train
prop = "formation_energy_per_atom"
datadir = "../MP_data/"
id = "task_id" # field by which to label each input material

# hyperparameters
num_conv = 3 # how many convolutional layers?
crys_fea_len = 32 # length of crystal feature vector after pooling
num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?

# atom featurization, pretty arbitrary choices for now
features = ["group", "row", "block", "atomic_mass", "atomic_radius", "X"]
num_bins = [18, 6, 4, 20, 10, 10]
num_features = sum(num_bins) # we'll use this later
logspaced = [false, false, false, true, true, false]
atom_feature_vecs = make_feature_vectors(features, num_bins, logspaced)

# dataset...first, read in outputs
info = CSV.read(string(datadir,prop,".csv"))
#y = Array(Float32.(info[!, Symbol(prop)]))

# shuffle data, pick out train/test
indices = shuffle(1:size(info,1))[1:num_pts]
info = info[indices,:]
#train_indices = indices[1:num_train]
#test_indices = indices[num_train+1:end]
#train_output = y[train_indices]
#test_output = y[test_indices]

# next, make graphs and build input features (matrices of dimension (# features, # nodes))
graphs = SimpleWeightedGraph[]
element_lists = Array{String}[]
inputs = Array{Bool}[]
for r in eachrow(info)
    cifpath = string(datadir, prop, "_cifs/", r[Symbol(id)], ".cif")
    graph, el_list = build_graph(cifpath)
    push!(graphs, graph)
    push!(element_lists, el_list)
    push!(inputs, vcat([atom_feature_vecs[e]' for e in el_list]...))
    # TODO: is this the best way to store this info?
end


# build the network
# 3 convolutional layers


# pooling function - mean

# then a softplus

# 1 fully-connected layer for output number

# then another softplus?

# then fully connected output to one value


# train
# ...
