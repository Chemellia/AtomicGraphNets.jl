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

cd(@__DIR__)

graph_dir = "../../../data/OCP/traj_test_graphs/"
bulk_graph_dir = "../../../data/OCP/traj_test_bulk_graphs/"

bulk_graphs_files = readdir(bulk_graph_dir, join=true)
surf_graphs_files = readdir(graph_dir, join=true)

# read in the graphs
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


