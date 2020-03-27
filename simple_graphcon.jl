#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#

using GraphPlot, Colors
using CSV
using SparseArrays
using Random, Statistics
using Flux
include("graph_functions.jl")
include("layers.jl")


# model hyperparameters – keeping it pretty simple for now
num_conv = 3 # how many convolutional layers?
pool_dims = (7,3) # pooling kernel size along features, nodes
pool_pad = (3,1)
pool_stride = (2,1)
crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
out_pool_features = Int64(floor((num_features+2*pool_pad[1]-pool_dims[1])/pool_stride[1] + 1))
opt = ADAM(0.001) # optimizer

# dataset...first, read in outputs
info = CSV.read(string(datadir,prop,".csv"))
y = Array(Float32.(info[!, Symbol(prop)]))

# shuffle data and pick out subset
indices = shuffle(1:size(info,1))[1:num_pts]
info = info[indices,:]
output = y[indices]

# next, make graphs and build input features (matrices of dimension (# features, # nodes))
graphs = SimpleWeightedGraph{Int32, Float32}[]
element_lists = Array{String}[]
inputs = Tuple{Array{Bool,2},SparseArrays.SparseMatrixCSC{Float32,Int64}}[]
for r in eachrow(info)
    cifpath = string(datadir, prop, "_cifs/", r[Symbol(id)], ".cif")
    graph, el_list = build_graph(cifpath)
    push!(graphs, graph)
    push!(element_lists, el_list)
    input = hcat([atom_feature_vecs[e] for e in el_list]...)
    push!(inputs, (input, adjacency_matrix(graph)))
end

# pick out train/test sets
train_output = y[1:num_train]
test_output = y[num_train+1:end]
train_input = inputs[1:num_train]
test_input = inputs[num_train+1:end]
train_data = zip(train_input, train_output)

# build the network (basically just copied from CGCNN.py for now): 3 convolutional layers, a mean pooling function, some dense layers, then fully connected output to one value for prediction

# TODO: make pooling less janky:
# * figure out pooling dimensionality thing... (for now just stuck those reshape/collapse layers in)
# * collapsing nodal dimension by a straight-up average right now, maybe need a custom layer that does these in one step?
#model = Chain([CGCNConv(num_features=>num_features) for i in 1:num_conv]..., x->x[1], x->reshape(x, (size(x)..., 1, 1)), MeanPool(pool_dims, stride=pool_stride, pad=pool_pad), x->mean(x, dims=2)[:,:,1,1], Dense(out_pool_features, crys_fea_len, softplus), [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers-1]..., Dense(crys_fea_len, 1, softplus))

# a way more simple example
model = Chain(CGCNConv(num_features=>num_features), x->x[1], x->reshape(x, (size(x)..., 1, 1)), x->mean(x))

# this works fine
@code_warntype model[4](model[3](model[2](model[1](inputs[1]))))

# but this gives an Any type...even though it should represent the same thing?
@code_warntype model(inputs[1])

# define loss function
#loss(x,y) = Flux.mse(model(x), y)

# train
#Flux.train!(loss, params(model), train_data, opt)
