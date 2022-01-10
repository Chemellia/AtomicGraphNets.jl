using Flux: glorot_uniform, normalise, @functor#, destructure
using Zygote: @adjoint, @nograd
using LinearAlgebra, SparseArrays
using Statistics
using SimpleWeightedGraphs
using DifferentialEquations, DiffEqSensitivity

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
(l::AGNConv)(a::FeaturizedAtoms{AtomGraph{T},GraphNodeFeaturization}) where T =
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

"""
Custom pooling layer that outputs a fixed-length feature vector irrespective of input dimensions, for consistent handling of different-sized graphs feeding to fully-connected dense layers afterwards. Adapted from Flux's MeanPool.

It accepts a pooling width and will adjust stride and/or padding such that the output vector length is correct.
"""
struct AGNPool
    pool_func::Function
    dim::Int64
    str::Int64
    pad::Int64
    function AGNPool(
        pool_type::String,
        in_num_features::Int64,
        out_num_features::Int64,
        pool_width_frac::Float64,
    )
        @assert in_num_features >= out_num_features "I don't think you actually want to pool to a LONGER vector, do you?"
        dim, str, pad =
            compute_pool_params(in_num_features, out_num_features, Float64(pool_width_frac))
        if pool_type == "max"
            pool_func = Flux.maxpool
        elseif pool_type == "mean"
            pool_func = Flux.meanpool
        end
        new(pool_func, dim, str, pad)
    end
end

pool_out_features(num_f::Int64, dim::Int64, stride::Int64, pad::Int64) =
    Int64(floor((num_f + 2 * pad - dim) / stride + 1))

"""
Helper function to work out dim, pad, and stride for desired number of output features, given a fixed pooling width.
"""
function compute_pool_params(
    num_f_in::Int64,
    num_f_out::Int64,
    dim_frac::AbstractFloat;
    start_dim = Int64(round(dim_frac * num_f_in)),
    start_str = Int64(floor(num_f_in / num_f_out)),
)
    # take starting guesses
    dim = start_dim
    str = start_str
    p_numer = str * (num_f_out - 1) - (num_f_in - dim)
    if p_numer < 0
        p_numer == -1 ? dim = dim + 1 : str = str + 1
    end
    p_numer = str * (num_f_out - 1) - (num_f_in - dim)
    if p_numer < 0
        error("problem, negative p!")
    end
    if p_numer % 2 == 0
        pad = Int64(p_numer / 2)
    else
        dim = dim - 1
        pad = Int64((str * (num_f_out - 1) - (num_f_in - dim)) / 2)
    end
    out_fea_len = pool_out_features(num_f_in, dim, str, pad)
    if !(out_fea_len == num_f_out)
        print("problem, output feature wrong length!")
    end
    # check if pad gets comparable to width...
    if pad >= 0.8 * dim
        @warn "specified pooling width was hard to satisfy without nonsensically large padding relative to width, had to increase from desired width"
        dim, str, pad = compute_pool_params(
            num_f_in,
            num_f_out,
            dim_frac,
            start_dim = Int64(round(1.2 * start_dim)),
        )
    end
    dim, str, pad
end

function (m::AGNPool)(feat::Matrix{<:Real})
    # compute what pad and stride need to be...
    x = reshape(feat, (size(feat)..., 1, 1))
    # do mean pooling across feature direction, average across all nodes in graph
    # TODO: decide if this approach makes sense or if there's a smarter way
    pdims = PoolDims(x, (m.dim, 1); padding = (m.pad, 0), stride = (m.str, 1))
    mean(m.pool_func(x, pdims), dims = 2)[:, :, 1, 1]
end

# alternate signatures so it can take output directly from AGNConv layer
(m::AGNPool)(lapl::Matrix{<:Real}, out_mat::Matrix{<:Real}) = m(out_mat)
(m::AGNPool)(t::Tuple{Matrix{R1},Matrix{R2}}) where {R1<:Real,R2<:Real} = m(t[2])

# DEQ-style model where we treat the convolution as a SteadyStateProblem
struct AGNConvDEQ{T,F}
    conv::AGNConv{T,F}
end

# NB can't do a pair of different numbers because otherwise get a size mismatch when trying to feed it back in...
function AGNConvDEQ(input_size::Integer, σ=softplus; initW=glorot_uniform, initb=zeros, T::DataType=Float32, bias::Bool=true)
    conv = AGNConv(input_size=>input_size, σ; initW=initW, initb=initb, T=T)
    AGNConvDEQ(conv)
end

@functor AGNConvDEQ

# set up SteadyStateProblem where the derivative is the convolution operation
# (we want the "fixed point" of the convolution)
# need it in the form f(u,p,t) (but t doesn't matter)
# u is the features, p is the parameters of conv
# re(p) reconstructs the convolution with new parameters p
function (l::AGNConvDEQ)(fa::FeaturizedAtoms)
    p, re = Flux.destructure(l.conv)
    # do one convolution to get initial guess
    guess = l.conv(fa)[2]

    f = function (dfeat,feat,p,t)
        input = fa.atoms.laplacian, reshape(feat,size(guess))
        output = re(p)(input)
        dfeat .= vec(output[2]) .- vec(input[2])
    end

    prob = SteadyStateProblem{true}(f, vec(guess), p)
    #return solve(prob, DynamicSS(Tsit5())).u
    alg = SSRootfind()
    #alg = SSRootfind(nlsolve = (f,u0,abstol) -> (res=SteadyStateDiffEq.NLsolve.nlsolve(f,u0,autodiff=:forward,method=:anderson,iterations=Int(1e6),ftol=abstol);res.zero))
    out_mat = reshape(
        solve(
            prob,
            alg,
            sensealg = SteadyStateAdjoint(autodiff = false, autojacvec = ZygoteVJP()),
        ).u,
        size(guess),
    )
    return fa.atoms.laplacian, out_mat
end

