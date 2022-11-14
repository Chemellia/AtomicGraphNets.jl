using AtomGraphs
using ChemistryFeaturization
using GeometricFlux
using AtomicGraphNets
import GraphSignals: normalized_adjacency_matrix
using Flux

# I literally pulled a random structure from Materials Project...(I did have to take out the oxidation state annotations)
ag = AtomGraph("Tl3VS4.cif")

# let's featurize it simply
fzn = GraphNodeFeaturization(["Block", "Atomic mass"])
fg = featurize(ag, fzn)
size(fg.encoded_features) # check size, should be 16 (# atoms) x 14 (# features, 4 for block and 10 for mass)

# here's a layer from AtomicGraphNets
la = AGNConv(14=>14)

# check that it works
la(fg) # first output is graph laplacian, second is new feature matrix

# we'll start simple, with GCNConv
# it has a syntax that takes in a normalized adjacency matrix and a feature matrix, so we can dispatch to that
# the only slight additional wrinkle is it expects the transpose in the argument, so we transpose input and then output again
(l::GCNConv)(fg::FeaturizedAtoms{AtomGraph{T}, GraphNodeFeaturization}) where T = Matrix(l(normalized_adjacency_matrix(fg.atoms.graph), fg.encoded_features')')

# let's also dispatch onto the tuples that AGNConv outputs so we can chain them
# note that this is a slight hack and will change the behavior a bit because we'll be passing a laplacian as an adjacency matrix
# also we transpose it back to "default" to the AGN convention as well as passing through the laplacian
(l::GCNConv)(t::Tuple{Matrix{R1},Matrix{R2}}) where {R1<:Real,R2<:Real} = (t[1], l(t[1], t[2]')')

# check that it works
lg = GCNConv(14=>10)
lg(fg)

# check also that we can chain them without issue
m = Chain(la, lg)
m(fg)

# but GCNConv is almost the same as AtomicGraphNets AGNConv, so it's not so interesting, let's try something a bit different...how about the `TopKPool` layer?
# (note that this layer as constructed takes a fixed adjacency matrix, so it's likely not that useful for our purposes, this is more for demonstration...though I think it would be pretty easy to make a version that doesn't need to be that way)
(tk::TopKPool)(t::Tuple{Matrix{R1},AbstractMatrix{R2}}) where {R1<:Real,R2<:Real} = tk(t[2]')

# check that it works
tk = TopKPool(fg.atoms.graph.weights, 8, 10) # adjacency matrix, k, input size
m = Chain(la, lg, tk)
m(fg)
