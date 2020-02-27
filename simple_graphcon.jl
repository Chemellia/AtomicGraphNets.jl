#using PyCall
using GeometricFlux
using SimpleWeightedGraphs, MetaGraphs
using GraphPlot, Colors

# from source code:
# (g::GCNConv)(X::AbstractMatrix) = g.σ.(g.weight * X * g.norm + g.bias)
# σ is identity so...
# tl(X) = tl.weight * X * tl.norm + tl.bias

adjmat = [[0. 8.]; [8. 0.]]
I = one(adjmat) # identity matrix
#norm = normalized_laplacian(adjmat+I)
# TODO: figure out what this should be - normalized or not? +I or not?
norm = laplacian_matrix(adjmat + I)
bias = zeros(2,2)

# NOTE: in Tian's SI example, it's a 3 => 1 convolution for feature length so you go straight to a scalar after one layer, but GCNConv lets us do whatever length we want. This is currently working analogously to the KCl example in the SI

# TODO: probably define custom layer origCGCNN or something  that explicitly has self weight, conv weight, and bias to be maximally similar to Tian's 

in_feat = [[1 0]; [0 1]] # maybe these should be concatenated?
wt = ones(2,2)
layer = GCNConv(wt, bias, norm, identity)

println(layer(in_feat))
