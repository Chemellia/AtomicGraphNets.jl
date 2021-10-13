using Flux
using Flux: glorot_uniform, normalise, @functor#, destructure
using Zygote: @adjoint, @nograd
using LinearAlgebra, SparseArrays
using Statistics
using ChemistryFeaturization
#using DifferentialEquations, DiffEqSensitivity

"""
    AGNConv(in=>out)

Atomic graph convolutional layer. Almost identical to GCNConv from GeometricFlux but adapted to be most similar to Tian's original AGNN structure, so explicitly has self and convolutional weights separately.

# Fields
- `selfweight::Array{T,2}`: weights applied to features at a node
- `convweight::Array{T,2}`: convolutional weights
- `bias::Array{T,2}`: additive bias (second dimension is always 1 because only learnable per-feature, not per-node)
- `σ::F`: activation function (will be applied before `reg_norm` to outputs), defaults to softplus

# Arguments
- `in::Integer`: the dimension of input features.
- `out::Integer`: the dimension of output features.
- `σ=softplus`: activation function
- `initW=glorot_uniform`: initialization function for weights
- `initb=zeros`: initialization function for biases

"""
struct AGNConv{T,F}
    selfweight::Array{T,2}
    convweight::Array{T,2}
    bias::Array{T,2}
    σ::F
end

function AGNConv(
    ch::Pair{<:Integer,<:Integer},
    σ = softplus;
    initW = glorot_uniform,
    initb = zeros,
    T::DataType = Float64,
)
    selfweight = T.(initW(ch[2], ch[1]))
    convweight = T.(initW(ch[2], ch[1]))
    b = T.(initb(ch[2], 1))
    AGNConv(selfweight, convweight, b, σ)
end

@functor AGNConv

"""
 Define action of layer on inputs: do a graph convolution, add this (weighted by convolutional weight) to the features themselves (weighted by self weight) and the per-feature bias (concatenated to match number of nodes in graph).

# Arguments
- input: a FeaturizedAtoms object, or graph_laplacian, encoded_features

# Note
In the case of providing two matrices, the following conditions must hold:
- `lapl` must be square and of dimension N x N where N is the number of nodes in the graph
- `X` (encoded features) must be of dimension M x N, where M is `size(l.convweight)[2]` (or equivalently, `size(l.selfweight)[2]`)
"""
function (l::AGNConv{T,F})(lapl::Matrix{<:Real}, X::Matrix{<:Real}) where {T<:Real,F}
    # should we put dimension checks here? Could allow more informative errors, but would likely introduce performance penalty. For now it's just in docstring.
    out_mat =
        T.(
            normalise(
                l.σ.(
                    l.convweight * X * lapl +
                    l.selfweight * X +
                    reduce(hcat, l.bias for i = 1:size(X, 2)),
                ),
                dims = [1, 2],
            ),
        )
    lapl, out_mat
end

# alternate signature so FeaturizedAtoms can be fed into first layer
(l::AGNConv)(a::FeaturizedAtoms{AtomGraph,GraphNodeFeaturization}) =
    l(a.atoms.laplacian, a.encoded_features)

# signature to splat appropriately
(l::AGNConv)(t::Tuple{Matrix{R1},Matrix{R2}}) where {R1<:Real,R2<:Real} = l(t...)

# fixes from Dhairya so backprop works
@adjoint function SparseMatrixCSC{T,N}(arr) where {T,N}
    SparseMatrixCSC{T,N}(arr), Δ -> (collect(Δ),)
end
@nograd LinearAlgebra.diagm

@adjoint function Broadcast.broadcasted(Float32, a::SparseMatrixCSC{T,N}) where {T,N}
    Float32.(a), Δ -> (nothing, T.(Δ))
end
@nograd issymmetric

@adjoint function Broadcast.broadcasted(Float64, a::SparseMatrixCSC{T,N}) where {T,N}
    Float64.(a), Δ -> (nothing, T.(Δ))
end

@adjoint function softplus(x::Real)
    y = softplus(x)
    return y, Δ -> (Δ * σ(x),)
end
