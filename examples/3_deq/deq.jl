#=
Basically the same as the first example, but trying the DEQ approach using SteadyStateProblem.
=#
#using Pkg
#Pkg.activate("../")
using CSV
using DataFrames
using Random, Statistics
using Flux
using Flux: @epochs
using SimpleWeightedGraphs
using ChemistryFeaturization
using AtomicGraphNets
using Serialization

println("Setting things up...")

# where to find the inputs
csv_path = "../2_qm9/qm9.csv"
xyz_dir = "../2_qm9/xyz/"
graph_dir = "../2_qm9/graphs/"

# data-related options
num_pts = 111
train_frac = 0.8 # what fraction for training?
num_epochs = 40 # how many epochs to train?
num_train = Int32(round(train_frac * num_pts))
num_test = num_pts - num_train
prop = :u0 # internal energy at 0K
info = DataFrame!(CSV.File(csv_path))
y = Array(Float32.(info[!, Symbol(prop)]))

# atom featurization, pretty arbitrary choices for now
features = Symbol.(["Group", "Row", "Block", "Atomic mass", "Atomic radius", "X"])
num_bins = [18, 9, 4, 16, 10, 10]
num_features = sum(num_bins) # we'll use this later
logspaced = [false, false, false, true, true, false]
atom_feature_vecs, featurization = make_feature_vectors(features, nbins=num_bins, logspaced=logspaced)

# model hyperparameters – keeping it pretty simple for now
crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
opt = ADAM(0.001) # optimizer

# shuffle data and pick out subset
indices = shuffle(1:size(info,1))[1:num_pts]
info = info[indices,:]
output = y[indices]

# next, make graphs and build input features (matrices of dimension (# features, # nodes))
println("Building graphs and feature vectors from structures...")
inputs = AtomGraph[]

#TODO: this with bulk processing fcn

for r in eachrow(info)
    graph_path = joinpath(graph_dir, string(r.mol_id, ".jls"))
    gr = deserialize(graph_path)
    feature_mat = hcat([atom_feature_vecs[e] for e in gr.elements]...)
    add_features!(gr, feature_mat, featurization)
    push!(inputs, gr)
end

# pick out train/test sets
println("Dividing into train/test sets...")
train_output = output[1:num_train]
test_output = output[num_train+1:end]
train_input = inputs[1:num_train]
test_input = inputs[num_train+1:end]
train_data = zip(train_input, train_output)

# moved DEQ to its own layer definition
model = Chain(AGNConvDEQ(num_features=>num_features), AGNMeanPool(crys_fea_len, 0.1), [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers]..., Dense(crys_fea_len, 1, softplus))

loss(x,y) = Flux.mse(model(x), y)
# and a callback to see training progress
evalcb() = @show(mean(loss.(test_input, test_output)))
evalcb() # this hangs for at least 20 minutes...

# TODO: troubleshoot training (type inference)
# --> maybe try a fixed graph with different features and only weights in the convolution
# so passing one matrix could maybe work?

# train
println("Training!")
#Flux.train!(loss, params(model), train_data, opt)
@epochs num_epochs Flux.train!(loss, params(model), train_data, opt, cb = Flux.throttle(evalcb, 5))

#=
# a much simpler case...one graph (just a triangle), just one feature per node
# only conv weight, no self weight or bias
f = function (dfeat,feat,convwt,t)
    gr = SimpleWeightedGraph{Int32, Float32}([0 1 1; 1 0 1; 1 1 0])
    output = softplus(convwt * feat * normalized_laplacian(gr) + feat)
    dfeat = output .- feat
end

struct SimpleDEQ{T}
    convwt::Array{T,2}
end

@functor SimpleDEQ

function SimpleDEQ(ch::Pair{<:Integer,<:Integer}; init=glorot_uniform, T::DataType=Float32, bias::Bool=true)
    convwt = init(ch[2], ch[1])
    SimpleDEQ(convwt)
end

function (l::SimpleDEQ)(gr::FeaturedGraph{T,S}) where {T,S}
    # do one convolution to get initial guess
    guess = l.convwt * feature(gr) * normalized_laplacian(graph(gr)) + feature(gr)
    prob = SteadyStateProblem{true}(f, guess, l.convwt)
    return solve(prob, DynamicSS(Tsit5())).u
end

layer = SimpleDEQ(1=>1)
poollyr = CGCNMeanPool(1, 0.2)
model = Chain(layer, poollyr)

gr = SimpleWeightedGraph{Int32, Float32}([0 1 1; 1 0 1; 1 1 0])

# try making some arbitrary training data
# just mapping to average for something stupid simple

feat1 = [3. 4. 5.]
out1 = 4.
inp1 = FeaturedGraph(gr, feat1)
feat2 = [0. 4. 5.]
out2 = 3.
inp2 = FeaturedGraph(gr, feat2)

simple_in = [inp1, inp2]
simple_out = [out1, out2]

# okay let's try it!
loss(x,y) = Flux.mse(model(x), y)
evalcb() = @show(mean(loss.(simple_in, simple_out)))
evalcb()

Flux.train!(loss, params(model), zip(simple_in, simple_out), opt)
=#
