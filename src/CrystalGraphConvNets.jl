module CrystalGraphConvNets

# should I put all the "using" statements that the whole package needs here?
import Flux:@functor

export inverse_square, exp_decay, build_graph, visualize_graph
include("graph_functions.jl")

export atom_data_df, make_feature_vectors, decode_feature_vector
include("featurize.jl")

export CGCNConv, CGCNMeanPool, CGCNMaxPool
include("layers.jl")

end
