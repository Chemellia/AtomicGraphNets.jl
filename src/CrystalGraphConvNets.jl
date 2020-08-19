module CrystalGraphConvNets

# should I put all the "using" statements that the whole package needs here?
import Flux:@functor

export CGCNConv, CGCNMeanPool, CGCNMaxPool
include("layers.jl")

end
