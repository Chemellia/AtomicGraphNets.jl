using Flux
using Flux: glorot_uniform, normalise, @functor#, destructure
using Zygote: @adjoint, @nograd
using LinearAlgebra
using ChemistryFeaturization

#= Pooling Layer based on
Graph Convolutional Networks with EigenPooling - Yao Ma, Suhang Wang, Charu C. Aggarwal, Jiliang Tang
https://arxiv.org/abs/1904.13107
=#

# TBD - what other fields would be necessary for the pooling layer itself?
struct EigenPool
    pool_func::Function
    function EigenPool()
        new(eigen_pool_func)
    end
end

function eigen_pool_func(adj_graph::Matrix{<:Real}, features::Matrix{<:Real})
    # graph_coarsening on adj_graph
    # apply EigenPooling operator on the pooled features to generate new node representations corresponding to coarsened graph
end

function (m::EigenPool)(adj_graph::Matrix{<:Real}, features::Matrix{<:Real})
    return m.pool_func(adj_graph, features)
end
