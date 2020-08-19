using Flux
using Flux: glorot_uniform, @functor
using Zygote: @adjoint, @nograd
using LinearAlgebra, SparseArrays
using GeometricFlux
using Statistics
using SimpleWeightedGraphs

# regularized norm fcn, cut out the dims part
function reg_norm(x::AbstractArray, ϵ=sqrt(eps(Float32)))
    μ′ = mean(x)
    σ′ = std(x, mean = μ′, corrected=false)
    return (x .- μ′) ./ (σ′ + ϵ)
end


struct CGCNConv{T,F}
    selfweight::Array{T,2}
    convweight::Array{T,2}
    bias::Array{T,2}
    σ::F
end

"""
    CGCNConv(in=>out)
    CGCNConv(in=>out, σ)

Crystal graph convolutional layer. Almost identical to GCNConv from GeometricFlux but adapted to be most similar to Tian's original CGCNN structure, so explicitly has self and convolutional weights separately. Default activation function is softplus.

# Arguments
- `in::Integer`: the dimension of input features.
- `out::Integer`: the dimension of output features.
- `σ::F=softplus`: activation function
- `bias::Bool=true`: keyword argument, whether to learn the additive bias.
"""
function CGCNConv(ch::Pair{<:Integer,<:Integer}, σ=softplus; initW=glorot_uniform, initb=zeros, T::DataType=Float32)
    selfweight = initW(ch[2], ch[1])
    convweight = initW(ch[2], ch[1])
    b = initb(ch[2], 1)
    CGCNConv(selfweight, convweight, b, σ)
end

@functor CGCNConv

# TODO here: in the case of chaining multiple of these layers together, should make a way to pass laplacian through so it doesn't have to get computed each time (maybe some kind of flag to specify which is being given?)
"""
 Define action of layer on inputs: do a graph convolution, add this (weighted by convolutional weight) to the features themselves (weighted by self weight) and the per-feature bias (concatenated to match number of nodes in graph).

# Arguments
- input: FeaturedGraph with  input data (stored in (# features, # nodes) order) and adjacency matrix of the graph
"""
#(l::CGCNConv)(input::Tuple{Array{Float32,2},SparseMatrixCSC{Float32,Int64}}) = l.σ.(l.convweight * input[1] * normalized_laplacian(input[2], Float32) + l.selfweight * input[1] + hcat([l.bias for i in 1:size(input[2], 1)]...)), input[2]

function (l::CGCNConv)(gr::FeaturedGraph{T,S}) where {T,S}
    X = feature(gr)
    A = graph(gr)
    out_mat = reg_norm(l.σ.(l.convweight * X * normalized_laplacian(A.weights, Float32) + l.selfweight * X + hcat([l.bias for i in 1:size(X, 2)]...)))
    FeaturedGraph(A, out_mat)
end

# alternate input format: adjacency matrix and feature matrix
(l::CGCNConv)(adjmat::AbstractMatrix{<:AbstractFloat}, fea::AbstractMatrix{<:AbstractFloat}) = l(FeaturedGraph(SimpleWeightedGraph(adjmat), fea))

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
Custom mean pooling layer that outputs a fixed-length feature vector irrespective of input dimensions, for consistent handling of different-sized graphs feeding to fully-connected dense layers afterwards. Adapted from Flux MeanPool.

It accepts a pooling width and will adjust stride and/or padding such that the output vector length is correct.
"""
struct CGCNMeanPool
    out_num_features::Int64
    pool_width_frac::Float32
end

pool_out_features(num_f::Int64, dim::Int64, stride::Int64, pad::Int64) = Int64(floor((num_f+2*pad-dim)/stride + 1))

"""
Helper function to work out dim, pad, and stride for desired number of output features, given a fixed pooling width.
"""
function compute_pool_params(num_f_in::Int64, num_f_out::Int64, dim_frac::Float32)
    # take starting guesses
    dim = Int64(round(dim_frac*num_f_in))
    str = Int64(floor(num_f_in/num_f_out))
    p_numer = str*(num_f_out-1) - (num_f_in - dim)
    if p_numer < 0
        if p_numer == -1
            dim = dim + 1
        else
            str = str + 1
        end
    end
    p_numer = str*(num_f_out-1) - (num_f_in - dim)
    if p_numer < 0
        print("problem, negative p!")
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
    dim, str, pad
end

function (m::CGCNMeanPool)(fg::FeaturedGraph{})
      # compute what pad and stride need to be...
      x = feature(fg)
      x = reshape(x, (size(x)..., 1, 1))
      num_features, num_nodes = size(x)
      dim, str, pad = compute_pool_params(num_features, m.out_num_features, m.pool_width_frac)
      # do mean pooling across feature direction, average across all nodes in graph
      # TODO: decide if this approach makes sense or if there's a smarter way
      pdims = PoolDims(x, (dim,1); padding=(pad,0), stride=(str,1))
      mean(Flux.meanpool(x, pdims), dims=2)[:,:,1,1]
end

"""Like above, but for max pooling"""
struct CGCNMaxPool
    out_num_features::Int64
    pool_width_frac::Float32
end

function (m::CGCNMaxPool)(fg::FeaturedGraph{})
      # compute what pad and stride need to be...
      x = feature(fg)
      x = reshape(x, (size(x)..., 1, 1))
      num_features, num_nodes = size(x)
      dim, str, pad = compute_pool_params(num_features, m.out_num_features, m.pool_width_frac)
      # do max pooling along feature direction, average across all nodes in graph
      # TODO: decide if this approach makes sense or if there's a smarter way
      pdims = PoolDims(x, (dim,1); padding=(pad,0), stride=(str,1))
      mean(Flux.maxpool(x, pdims), dims=2)[:,:,1,1]
end
