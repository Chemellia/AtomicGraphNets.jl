module AtomicGraphNets

export AGNConv, AGNPool, AGNConvDEQ
include("layers.jl")
using .Layers: AGNConv, AGNPool, AGNConvDEQ

include("models.jl")
export build_CGCNN, build_SGCNN, build_CGCNN_DEQ

end
