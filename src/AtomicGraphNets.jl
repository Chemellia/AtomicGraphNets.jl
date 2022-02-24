module AtomicGraphNets

using AtomGraphs
using ChemistryFeaturization
using Flux

include("graphnodefeaturization.jl")
export GraphNodeFeaturization, encode, decode, features
export encodable_elements, chunk_vec, output_shape

export AGNConv, AGNPool, AGNConvDEQ
include("layers.jl")

include("models.jl")
export build_CGCNN, build_SGCNN, build_CGCNN_DEQ

end
