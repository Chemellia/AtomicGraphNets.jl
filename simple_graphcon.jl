using PyCall
using GeometricFlux
using LightGraphs, SimpleWeightedGraphs, MetaGraphs
using GraphPlot, Colors

# from source code:
# (g::GCNConv)(X::AbstractMatrix) = g.σ.(g.weight * X * g.norm + g.bias)
# σ is identity so...
# tl(X) = tl.weight * X * tl.norm + tl.bias

# make a connected line of four nodes
adjmat = [[0. 1. 0. 0.]; [1. 0. 1. 0.]; [0. 1. 0. 1.]; [0. 0. 1. 0.]]
I = one(adjmat) # identity matrix
norm = normalized_laplacian(adjmat+I)
bias = zeros(1,4)
wt = ones(1,1)

layer = GCNConv(wt, bias, norm, identity)

in_feat = [1. 2. 3. 4.]

println(layer(in_feat))
