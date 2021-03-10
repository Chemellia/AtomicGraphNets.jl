#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#
using CSV, DataFrames
using Random, Statistics
using Flux
using Flux: @epochs
using SimpleWeightedGraphs
using ChemistryFeaturization
using AtomicGraphNets

println("Setting things up...")

# data-related options
num_pts = 100 # how many points to use?
train_frac = 0.8 # what fraction for training?
num_epochs = 5 # how many epochs to train?
num_train = Int32(round(train_frac * num_pts))
num_test = num_pts - num_train
prop = "formation_energy_per_atom"
datadir = "../../MP_data/"
id = "task_id" # field by which to label each input material

# atom featurization, pretty arbitrary choices for now
features = Symbol.(["Group", "Row", "Block", "Atomic mass", "Atomic radius", "X"])
num_bins = [18, 9, 4, 16, 10, 10]
num_features = sum(num_bins) # we'll use this later
logspaced = [false, false, false, true, true, false]
# returns actual vectors (in a dict with keys of elements) plus Vector of AtomFeat objects describing featurization metadata
atom_feature_vecs, featurization = make_feature_vectors(features, nbins=num_bins, logspaced=logspaced)

# model hyperparameters – keeping it pretty simple for now
num_conv = 3 # how many convolutional layers?
crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
opt = ADAM(0.001) # optimizer

# dataset...first, read in outputs
info = CSV.read(string(datadir,prop,".csv"), DataFrame)
y = Array(Float32.(info[!, Symbol(prop)]))

# shuffle data and pick out subset
indices = shuffle(1:size(info,1))[1:num_pts]
info = info[indices,:]
output = y[indices]

# next, make graphs and build input features (matrices of dimension (# features, # nodes))
println("Building graphs and feature vectors from structures...")
inputs = AtomGraph[]

for r in eachrow(info)
    cifpath = string(datadir, prop, "_cifs/", r[Symbol(id)], ".cif")
    gr = build_graph(cifpath)
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
model = Xie_model(num_features, num_conv=num_conv, atom_conv_feature_length=crys_fea_len, num_hidden_layers=1)

# define loss function and a callback to monitor progress
loss(x,y) = Flux.mse(model(x), y)
evalcb() = @show(mean(loss.(test_input, test_output)))
evalcb()

# train
println("Training!")
@epochs num_epochs Flux.train!(loss, params(model), train_data, opt, cb = Flux.throttle(evalcb, 5))
