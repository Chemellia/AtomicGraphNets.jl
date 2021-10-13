using Flux
using Flux: glorot_uniform, normalise, @functor#, destructure
using Zygote: @adjoint, @nograd
using LinearAlgebra
using ChemistryFeaturization
#using DifferentialEquations, DiffEqSensitivity

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
