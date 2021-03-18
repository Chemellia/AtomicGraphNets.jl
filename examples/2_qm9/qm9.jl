#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#
using CSV, DataFrames
using Random, Statistics
using Flux
using Flux: @epochs
using ChemistryFeaturization
using AtomicGraphNets
using Serialization

cd(@__DIR__)

println("Setting things up...")

# where to find the inputs
csv_path = "qm9.csv"
xyz_dir = "xyz/"
graph_dir = "graphs/"

# data-related options
num_pts = 111
train_frac = 0.8 # what fraction for training?
num_epochs = 40 # how many epochs to train?
num_train = Int32(round(train_frac * num_pts))
num_test = num_pts - num_train
prop = :u0 # internal energy at 0K
info = DataFrame!(CSV.File(csv_path))
y = Array(Float32.(info[!, Symbol(prop)]))

# atom featurization, pretty arbitrary choices for now
features = Symbol.(["Group", "Row", "Block", "Atomic mass", "Atomic radius", "X"])
num_bins = [18, 9, 4, 16, 10, 10]
num_features = sum(num_bins) # we'll use this later
logspaced = [false, false, false, true, true, false]
# returns actual vectors (in a dict with keys of elements) plus Vector of AtomFeat objects describing featurization metadata
atom_feature_vecs, featurization = make_feature_vectors(features, nbins=num_bins, logspaced=logspaced)

# model hyperparameters – keeping it pretty simple for now
num_conv = 5 # how many convolutional layers?
atom_fea_len = 80
crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
pool_type = "max"
num_hidden_layers = 2 # how many fully-connected layers after convolution and pooling?
opt = ADAM(0.003) # optimizer

# shuffle data and pick out subset
indices = shuffle(1:size(info,1))[1:num_pts]
info = info[indices,:]
output = y[indices]

# next, make graphs and build input features (matrices of dimension (# features, # nodes))
println("Building graphs and feature vectors from structures...")
inputs = AtomGraph[]

# build the graphs
build_graphs_batch(xyz_dir; output_folder = graph_dir)

for r in eachrow(info)
    graph_path = joinpath(graph_dir, string(r.mol_id, ".jls"))
    gr = deserialize(graph_path)
    feature_mat = hcat([atom_feature_vecs[e] for e in gr.elements]...)
    add_features!(gr, feature_mat, featurization)
    push!(inputs, gr)
end

# pick out train/test sets
println("Dividing into train/test sets...")
train_output = output[1:num_train]
test_output = output[num_train+1:end]
train_input = inputs[1:num_train]
test_input = inputs[num_train+1:end]
train_data = zip(train_input, train_output)

# build the model
println("Building the network...")
model = Xie_model(num_features, num_conv=num_conv, atom_conv_feature_length=atom_fea_len, pool_type=pool_type, pooled_feature_length=crys_fea_len, num_hidden_layers=num_hidden_layers)

# define loss function
loss(x,y) = Flux.mse(model(x), y)
# and a callback to see training progress
evalcb() = @show(mean(loss.(test_input, test_output)))
evalcb()

# train
println("Training!")
@epochs num_epochs Flux.train!(loss, params(model), train_data, opt, cb = Flux.throttle(evalcb, 10))
