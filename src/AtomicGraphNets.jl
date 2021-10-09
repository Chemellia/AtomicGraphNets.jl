module AtomicGraphNets

export AGNConv, AGNPool#, AGNConvDEQ
include("layers/layers.jl")
using .Layers: AGNConv, AGNPool

include("models.jl")
export build_CGCNN, build_SGCNN

end
