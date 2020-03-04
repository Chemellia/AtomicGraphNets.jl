using Flux
using Flux: glorot_uniform
using Flux: @functor
using GeometricFlux

# next bits based heavily off of https://github.com/yuehhua/GeometricFlux.jl/blob/master/src/layers/conv.jl

# this is probably closest to the "old" version in Tian's paper, but I want to train it and see what happens...
"""
    CGCNConv(graph, in=>out)
    CGCNConv(graph, in=>out, σ)
Crystal Graph convolutional layer.
# Arguments
- `graph`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `in`: the dimension of input features.
- `out`: the dimension of output features.
- `σ`: activation function
Data should be stored in (# features, # nodes) order.
For example, a 1000-node graph each node of which poses 100 features is constructed.
The input data would be a `1000×100` array.
"""
struct CGCNConv{T,F}
    selfweight::AbstractMatrix{T}
    convweight::AbstractMatrix{T}
    bias::AbstractMatrix{T}
    norm::AbstractMatrix{T}
    σ::F
end

function CGCNConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = size(adj, 1)
    b = init(ch[2], N)
    CGCNConv(init(ch[2], ch[1]), init(ch[2], ch[1]), b, normalized_laplacian(adj, T), σ)
end

@functor GCNConv

(g::GCNConv)(X::AbstractMatrix) = g.σ.(g.convweight * X * g.norm + g.selfweight * X + g.bias)
