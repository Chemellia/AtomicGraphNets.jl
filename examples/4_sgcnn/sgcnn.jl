#=
Toy version of slab graph networks, a la Kim et al. (dx.doi.org/10.1021/acs.chemmater.9b03686)

Lots of hacky stuff at the moment that ideally needs to get cleaned up...
=#

using Flux
using Flux: @epochs
using Random
using ChemistryFeaturization
using AtomicGraphNets
using Serialization
<<<<<<< HEAD

#cd(@__DIR__)

graph_dir = "../../../data/OCP/traj_test_graphs/"
bulk_graph_dir = "../../../data/OCP/traj_test_bulk_graphs/"

bulk_graphs_files = readdir(bulk_graph_dir, join=true)
surf_graphs_files = readdir(graph_dir, join=true)

# read in the graphs
=======
using CSV, DataFrames
using Statistics

#cd(@__DIR__)

# paths, options
graph_dir = "/Users/rkurchin/GDrive/CMU/research/ARPA-E/data/OCP/traj_test_graphs/"
bulk_graph_dir = "/Users/rkurchin/GDrive/CMU/research/ARPA-E/data/OCP/traj_test_bulk_graphs/"
csv_path = "/Users/rkurchin/GDrive/CMU/research/ARPA-E/data/OCP/10k_train.csv"
num_pts = 100
train_frac = 0.8
num_epochs = 5
opt = ADAM(0.001) # optimizer

num_train = Int32(round(train_frac * num_pts))
num_test = num_pts - num_train

# read in labels
info = CSV.File(csv_path) |> DataFrame
y = Array(Float32.(info[!, Symbol("energy")]))

# and the graphs
bulk_graphs_files = readdir(bulk_graph_dir, join=true)
surf_graphs_files = readdir(graph_dir, join=true)

bulk_graphs = read_graphs_batch(bulk_graph_dir)
surf_graphs = read_graphs_batch(graph_dir)

# pick out the indices for which we have bulk graphs constructed
keep_inds = []
for i in 1:length(surf_graphs_files)
    fn = splitpath(surf_graphs_files[i])[end]
    if isfile(joinpath(bulk_graph_dir, fn)) && fn[end-3:end]==".jls"
        append!(keep_inds, [i])
    end
end
surf_graphs = surf_graphs[keep_inds]
info = info[keep_inds, :]
y = y[keep_inds]

keep_inds = []
# now cut out any with NaN laplacians in either set
for i in 1:length(bulk_graphs)
    
end

# shuffle data and pick out subset
indices = shuffle(1:size(info,1))[1:num_pts]
info = info[indices, :]
output = y[indices]
bulk_graphs = bulk_graphs[indices]
surf_graphs = surf_graphs[indices]

# atom featurization, pretty arbitrary choices for now
features = Symbol.(["Group", "Row", "Block", "Atomic mass", "Atomic radius", "X"])
num_bins = [18, 9, 4, 16, 10, 10]
num_features = sum(num_bins) # we'll use this later
logspaced = [false, false, false, true, true, false]
atom_feature_vecs, featurization = make_feature_vectors(features, nbins=num_bins, logspaced=logspaced)

# add the features to the graphs
for ag in surf_graphs
    add_features!(ag, atom_feature_vecs, featurization)
end
for ag in bulk_graphs
    add_features!(ag, atom_feature_vecs, featurization)
end

# now "tuple them up"
@assert length(surf_graphs)==length(bulk_graphs) "List lengths don't match up, something has gone wrong! :("

inputs = zip(bulk_graphs, surf_graphs)

inputs = [p for p in zip(bulk_graphs, surf_graphs)]

# pick out train/test sets
println("Dividing into train/test sets...")
train_output = output[1:num_train]
test_output = output[num_train+1:end]
train_input = inputs[1:num_train]
test_input = inputs[num_train+1:end]
train_data = zip(train_input, train_output)

# define model, loss, etc.
model = build_SGCNN(num_features)
loss(x,y) = Flux.mse(model(x), y)
evalcb() = @show(mean(loss.(test_input, test_output)))
evalcb()

# train
println("Training!")
#Flux.train!(loss, params(model), train_data, opt)
@epochs num_epochs Flux.train!(loss, params(model), train_data, opt, cb = Flux.throttle(evalcb, 5))
