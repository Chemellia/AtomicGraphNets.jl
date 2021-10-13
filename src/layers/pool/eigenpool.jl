using Flux
using Flux: glorot_uniform, normalise, @functor#, destructure
using Zygote: @adjoint, @nograd
using LinearAlgebra
using ChemistryFeaturization

#= Pooling Layer based on
Graph Convolutional Networks with EigenPooling - Yao Ma, Suhang Wang, Charu C. Aggarwal, Jiliang Tang
https://arxiv.org/abs/1904.13107

Since the size of the matrices we are dealing with isn't as huge as the one in the original work,
let us try using this as a global pooling mechanism instead, i.e., the adjacency matrix of the crystal
itself would be only "subgraph" from which we "coarsen" to a single node and pool accordingly, which
in theory would give us the overall graph representation
=#

# TBD - what other fields would be necessary for the pooling layer itself?
struct EigenPool
    pool::Function # pooling operator applied over the adjacency matrix
    out_feature_size::Int64
    function EigenPool(out_feature_size::Int64)
        new(eigen_pooling, out_feature_size)
    end
end

# here, L is the laplacian matrix
# this probably needs to be optimized.
function eigen_pooling(L::Matrix{<:Real}, features::Matrix{<:Real}, out_feature_size::Int64)
    L_eigen_vectors = eigvecs(L)    # find eigen vectors for L

    N = size(L_eigen_vectors)[1] # graph size
    d = Integer(length(features)/N) # dimensions of each feature vector

    H = Integer(floor(out_feature_size/d)) # number of features to be pooled
    H = H > N ? N : H # if H is greater than the number of nodes, then we pool all the nodes

    pad_len = out_feature_size - (d * H)

    result = Vector()

    for i = 1:H
        L_i = L_eigen_vectors[:, i]
        push!(result, L_i'*features)
    end

    # return H features + zero padded elements hcatt-ed into a single 1xdH matrix
    result = hcat(result..., zeros(Float64, pad_len, 1)...)'
    reshape(result, length(result), 1)  # return it as a dHx1 Matrix
end

#=
 = Given an adjacency graph and the corresponding matrix of node features, return the
 = pooled node features which is the final graph representation fed to the dense layer
 = The adj_graph would be `FeaturizedAtoms.atoms` and features would be the same as usual
 =#
(m::egnpl)(adj_graph::Matrix{<:Real}, features::Matrix{<:Real}) = egnpl.pool(adj_graph, features, m.out_feature_size)
