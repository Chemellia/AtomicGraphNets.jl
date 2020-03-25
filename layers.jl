using Flux
using Flux: glorot_uniform
using Flux: @functor
using GeometricFlux

# next bits based heavily off of https://github.com/yuehhua/GeometricFlux.jl/blob/master/src/layers/conv.jl

# this is probably closest to the "old" version in Tian's paper, but I want to train it and see what happens...
"""
    CGCNConv(graph, in=>out)
    CGCNConv(graph, in=>out, σ)

Crystal graph convolutional layer. Almost identical to GCNConv from GeometricFlux but adapted to be most similar to Tian's original CGCNN structure, so explicitly has self and convolutional weights separately.

# Arguments
- `graph`: either an adjacency matrix or a SimpleWeightedGraph.
- `in`: the dimension of input features.
- `out`: the dimension of output features.
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.

Data should be stored in (# features, # nodes) order.
For example, if the graph has 10 nodes, each of which has a feature vector of length 100, input data should be of dimension `10 x 100`.
"""
struct CGCNConv{T,F}
    selfweight::AbstractMatrix{T}
    convweight::AbstractMatrix{T}
    bias::AbstractMatrix{T}
    norm::AbstractMatrix{T}
    σ::F
end


function CGCNConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, σ = identity; init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = size(adj, 1)
    b = init(ch[2], N)
    CGCNConv(init(ch[2], ch[1]), init(ch[2], ch[1]), b, normalized_laplacian(adj, T), σ) # don't add identity in here because we're doing self weight separately
end

@functor CGCNConv

(g::CGCNConv)(X::AbstractMatrix) = g.σ.(g.convweight * X * g.norm + g.selfweight * X + g.bias)
