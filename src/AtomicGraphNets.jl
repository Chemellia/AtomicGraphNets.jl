module AtomicGraphNets

# should I put all the "using" statements that the whole package needs here?
import Flux:@functor

include("layers.jl")
export AGNConv, AGNMeanPool, AGNMaxPool

include("models.jl")
export Xie_model

end
