using Flux
using Flux: glorot_uniform, @functor
using Zygote: @adjoint, @nograd
using LinearAlgebra, SparseArrays
using Statistics
using SimpleWeightedGraphs
using ChemistryFeaturization

# regularized norm fcn, cut out the dims part
function reg_norm(x::AbstractArray, ϵ=sqrt(eps(Float32)))
    μ′ = mean(x)
    σ′ = std(x, mean = μ′, corrected=false)
    return Float32.((x .- μ′) ./ (σ′ + ϵ))
end


struct AGNConv{T,F}
    selfweight::Array{T,2}
    convweight::Array{T,2}
    bias::Array{T,2}
    σ::F
end

"""
    AGNConv(in=>out)
    AGNConv(in=>out, σ)

Atomic graph convolutional layer. Almost identical to GCNConv from GeometricFlux but adapted to be most similar to Tian's original AGNN structure, so explicitly has self and convolutional weights separately. Default activation function is softplus.

# Arguments
- `in::Integer`: the dimension of input features.
- `out::Integer`: the dimension of output features.
- `σ::F=softplus`: activation function
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.
"""
function AGNConv(ch::Pair{<:Integer,<:Integer}, σ=softplus; initW=glorot_uniform, initb=zeros, T::DataType=Float32)
    selfweight = T.(initW(ch[2], ch[1]))
    convweight = T.(initW(ch[2], ch[1]))
    b = T.(initb(ch[2], 1))
    AGNConv(selfweight, convweight, b, σ)
end

@functor AGNConv

"""
 Define action of layer on inputs: do a graph convolution, add this (weighted by convolutional weight) to the features themselves (weighted by self weight) and the per-feature bias (concatenated to match number of nodes in graph).

# Arguments
- input: AtomGraph object
"""
function (l::AGNConv)(ag::AtomGraph)
    lapl = ag.lapl
    X = ag.features
    out_mat = Float32.(reg_norm(l.σ.(l.convweight * X * lapl + l.selfweight * X + hcat([l.bias for i in 1:size(X, 2)]...))))
    AtomGraph(ag.graph, ag.elements, ag.lapl, out_mat, AtomFeat[])
end

# fixes from Dhairya so backprop works
@adjoint function SparseMatrixCSC{T,N}(arr) where {T,N}
  SparseMatrixCSC{T,N}(arr), Δ -> (collect(Δ),)
end
@nograd LinearAlgebra.diagm

@adjoint function Broadcast.broadcasted(Float32, a::SparseMatrixCSC{T,N}) where {T,N}
  Float32.(a), Δ -> (nothing, T.(Δ), )
end
@nograd issymmetric

@adjoint function softplus(x::Real)
  y = softplus(x)
  return y, Δ -> (Δ * σ(x),)
end

"""
Custom pooling layer that outputs a fixed-length feature vector irrespective of input dimensions, for consistent handling of different-sized graphs feeding to fully-connected dense layers afterwards. Adapted from Flux MeanPool.

It accepts a pooling width and will adjust stride and/or padding such that the output vector length is correct.
"""
struct AGNPool
    pool_func::Function
    dim::Int64
    str::Int64
    pad::Int64
    function AGNPool(pool_type::String, in_num_features::Int64, out_num_features::Int64, pool_width_frac::AbstractFloat)
    dim, str, pad = compute_pool_params(in_num_features, out_num_features, Float32(pool_width_frac))
    if pool_type=="max"
        pool_func = Flux.maxpool
    elseif pool_type=="mean"
        pool_func = Flux.meanpool
    end
    new(pool_func, dim, str, pad)
    end
end

pool_out_features(num_f::Int64, dim::Int64, stride::Int64, pad::Int64) = Int64(floor((num_f + 2 * pad - dim) / stride + 1))

"""
Helper function to work out dim, pad, and stride for desired number of output features, given a fixed pooling width.
"""
function compute_pool_params(num_f_in::Int64, num_f_out::Int64, dim_frac::Float32; start_dim=Int64(round(dim_frac*num_f_in)), start_str=Int64(floor(num_f_in/num_f_out)))
    # take starting guesses
    dim = start_dim
    str = start_str
    p_numer = str*(num_f_out-1) - (num_f_in - dim)
    if p_numer < 0
        p_numer == -1 ? dim = dim + 1 : str = str + 1
    end
    p_numer = str*(num_f_out-1) - (num_f_in - dim)
    if p_numer < 0
        error("problem, negative p!")
    end
    if p_numer % 2 == 0
        pad = Int64(p_numer/2)
    else
        dim = dim - 1
        pad = Int64((str*(num_f_out-1) - (num_f_in - dim))/2)
    end
    out_fea_len = pool_out_features(num_f_in, dim, str, pad)
    if !(out_fea_len==num_f_out)
        print("problem, output feature wrong length!")
    end
    # check if pad gets comparable to width...
    if pad >= 0.8*dim
        @warn "specified pooling width was hard to satisfy without nonsensically large padding relative to width, had to increase from desired width"
        dim, str, pad  = compute_pool_params(num_f_in, num_f_out, dim_frac, start_dim=Int64(round(1.2*start_dim)))
    end
    dim, str, pad
end

function (m::AGNPool)(ag::AtomGraph)
      # compute what pad and stride need to be...
      x = ag.features
      x = reshape(x, (size(x)..., 1, 1))
      # do mean pooling across feature direction, average across all nodes in graph
      # TODO: decide if this approach makes sense or if there's a smarter way
      pdims = PoolDims(x, (m.dim,1); padding=(m.pad,0), stride=(m.str,1))
      mean(m.pool_func(x, pdims), dims=2)[:,:,1,1]
end
