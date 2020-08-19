#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#
using Pkg
Pkg.activate("../../")
#using GraphPlot, Colors
using CSV
using SparseArrays
using Random, Statistics
using Flux
using Flux: @epochs
using GeometricFlux: FeaturedGraph
using SimpleWeightedGraphs
using CrystalGraphConvNets
using ChemistryFeaturization

println("Setting things up...")

# data-related options
num_pts = 100 # how many points to use? Up to 32530 in the formation energy case as of 2020/04/01
train_frac = 0.8 # what fraction for training?
num_epochs = 5 # how many epochs to train?
num_train = Int32(round(train_frac * num_pts))
num_test = num_pts - num_train
prop = "formation_energy_per_atom"
datadir = "../../data/"
id = "task_id" # field by which to label each input material

# atom featurization, pretty arbitrary choices for now
features = Symbol.(["Group", "Row", "Block", "Atomic mass", "Atomic radius", "X"])
num_bins = [18, 9, 4, 16, 10, 10]
num_features = sum(num_bins) # we'll use this later
logspaced = [false, false, false, true, true, false]
atom_feature_vecs = make_feature_vectors(features, num_bins, logspaced)

# model hyperparameters – keeping it pretty simple for now
num_conv = 3 # how many convolutional layers?
crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
opt = ADAM(0.001) # optimizer

# dataset...first, read in outputs
info = CSV.read(string(datadir,prop,".csv"))
y = Array(Float32.(info[!, Symbol(prop)]))

# shuffle data and pick out subset
indices = shuffle(1:size(info,1))[1:num_pts]
info = info[indices,:]
output = y[indices]

# next, make graphs and build input features (matrices of dimension (# features, # nodes))
println("Building graphs and feature vectors from structures...")
#graphs = SimpleWeightedGraph{Int32, Float32}[]
element_lists = Array{String}[]
#inputs = Tuple{Array{Float32,2},SparseArrays.SparseMatrixCSC{Float32,Int64}}[]
inputs = FeaturedGraph{SimpleWeightedGraph{Int32, Float32}, Array{Float32,2}}[]
# TODO: figure out null pyobject issue with build_graph
for r in eachrow(info)
    cifpath = string(datadir, prop, "_cifs/", r[Symbol(id)], ".cif")
    gr, els = build_graph(cifpath)
    #push!(graphs, graph)
    push!(element_lists, els)
    #input = hcat([atom_feature_vecs[e] for e in el_list]...)
    feature_mat = hcat([atom_feature_vecs[e] for e in els]...)
    input = FeaturedGraph(SimpleWeightedGraph{Int32}(Float32.(gr)), feature_mat)
    #push!(inputs, (input, adjacency_matrix(graph)))
    push!(inputs, input)
end

# pick out train/test sets
println("Dividing into train/test sets...")
train_output = output[1:num_train]
test_output = output[num_train+1:end]
train_input = inputs[1:num_train]
test_input = inputs[num_train+1:end]
train_data = zip(train_input, train_output)

# build the network (basically just copied from CGCNN.py for now): the convolutional layers, a mean pooling function, some dense layers, then fully connected output to one value for prediction

println("Building the network...")
model = Chain([CGCNConv(num_features=>num_features) for i in 1:num_conv]..., CGCNMeanPool(crys_fea_len, 0.1), [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers]..., Dense(crys_fea_len, 1, softplus))

# TODO: MaxPool might make more sense

# define loss function
loss(x,y) = Flux.mse(model(x), y)
# and a callback to see training progress
evalcb() = @show(mean(loss.(test_input, test_output)))
evalcb()

# train
println("Training!")
#Flux.train!(loss, params(model), train_data, opt)
@epochs num_epochs Flux.train!(loss, params(model), train_data, opt, cb = Flux.throttle(evalcb, 5))
