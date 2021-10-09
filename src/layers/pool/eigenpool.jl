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
    function EigenPool()
        new(eigen_pooling)
    end
end

function eigen_pooling(graph::Matrix{<:Real}, features::Matrix{<:Real})
    L = Atoms.normalized_laplacian(graph) # get the laplacian matrix
    L_eigen_vectors = eigvecs(L)    # find eigen vectors for L
    result = Vector()

    for i = 1:size(L_eigen_vectors)[1]
        L_i = L_eigen_vectors[:, i]
        push!(result, L_i'*features)
    end

    # using an agreeable H and then return H elements of result hcatt-ed into a single 1xdH vector

end

#=
 = Given an adjacency graph and the corresponding matrix of node features, return the
 = pooled node features which is the final graph representation fed to the dense layer
 = The adj_graph would be `FeaturizedAtoms.atoms` and features would be the same as usual
 =#
function (m::EigenPool)(adj_graph::Matrix{<:Real}, features::Matrix{<:Real})
    features = m.pool(adj_graph, features)
    return adj_graph, features
end
