#using DifferentialEquations, DiffEqSensitivity

include("conv/agnconv.jl")
include("pool/agnpool.jl")

# following commented out for now because it only runs suuuuper slowly but slows down precompilation a lot
"""
# DEQ-style model where we treat the convolution as a SteadyStateProblem
struct AGNConvDEQ{T,F}
    conv::AGNConv{T,F}
end

function AGNConvDEQ(ch::Pair{<:Integer,<:Integer}, σ=softplus; initW=glorot_uniform, initb=glorot_uniform, T::DataType=Float32, bias::Bool=true)
    conv = AGNConv(ch, σ; initW=initW, initb=initb, T=T)
    AGNConvDEQ(conv)
end

@functor AGNConvDEQ

# set up SteadyStateProblem where the derivative is the convolution operation
# (we want the "fixed point" of the convolution)
# need it in the form f(u,p,t) (but t doesn't matter)
# u is the features, p is the parameters of conv
# re(p) reconstructs the convolution with new parameters p
function (l::AGNConvDEQ)(fa::FeaturizedAtoms)
    p,re = Flux.destructure(l.conv)
    # do one convolution to get initial guess
    guess = l.conv(gr)[2]

    f = function (dfeat,feat,p,t)
        input = gr
        input.encoded_features = reshape(feat,size(guess))
        output = re(p)(input)
        dfeat .= vec(output[2]) .- vec(input.encoded_features)
    end

    prob = SteadyStateProblem{true}(f, vec(guess), p)
    #return solve(prob, DynamicSS(Tsit5())).u
    alg = SSRootfind()
    #alg = SSRootfind(nlsolve = (f,u0,abstol) -> (res=SteadyStateDiffEq.NLsolve.nlsolve(f,u0,autodiff=:forward,method=:anderson,iterations=Int(1e6),ftol=abstol);res.zero))
    out_mat = reshape(solve(prob, alg, sensealg = SteadyStateAdjoint(autodiff = false, autojacvec = ZygoteVJP())).u,size(guess))
    return AtomGraph(gr.graph, gr.elements, out_mat, gr.featurization)
end
"""
