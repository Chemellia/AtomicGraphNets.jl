module AtomicGraphNets

export AGNConv, AGNPool#, AGNConvDEQ
include("layers.jl")
using .layers: AGNConv, AGNPool

include("models.jl")
export build_CGCNN, build_SGCNN

end
