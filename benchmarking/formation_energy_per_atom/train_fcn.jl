#=
 Initial hyperparameter optimization for comparison of CrystalGraphConvNets.jl to CGCNN.py.
=#
using Pkg
Pkg.activate("/home/rkurchin/CrystalGraphConvNets.jl/")
using CSV
using SparseArrays
using Random, Statistics
using Flux
using Flux: @epochs
using GeometricFlux
using SimpleWeightedGraphs
using CrystalGraphConvNets
using DelimitedFiles
using Distributed

"""
Playing with hyperparameters for CGCNN to predict formation energies.

Arguments:
    `num_pts::Integer`: how many structures to train on (up to 32530 currently)
    `num_conv::Integer`: how many convolutional layers
    `atom_fea_len::Integer`: how many atom features to reduce to after first convolution
    `pool_type::String`: max or mean
    `cry_fea_len::Integer`: number of crystal features to pool to after convolution
    `num_hidden_layers::Integer`: how many fully-connected layers after pooling
    `lr::Float32`: Learning rate for optimizer
    `features::Array{String,1}`: what features to include
    `num_bins::Array{Integer,1}`: how many bins for each feature
    `logspaced::Array{Bool,1}`: spacing for each feature (false for linear, true for log)
    `cutoff_radius::Float32`: cutoff radius for an atom to be considered a neighbor when building graph (angstroms)
    `nbr_num_cutoff::Integer`: (soft) maximum neighbor number to include
    `decay_fcn`: fcn describing decay of graph weight with interatomic distance
"""
#function cgcnn_train(num_pts, num_conv, atom_fea_len, pool_type, crys_fea_len, num_hidden_layers, lr, features, num_bins, logspaced, cutoff_radius=8.0, max_num_nbr=12, decay_fcn=inverse_square)
function cgcnn_train(args)
    num_pts, num_conv, atom_fea_len, pool_type, crys_fea_len, num_hidden_layers, lr, features, num_bins, logspaced = args
    cutoff_radius=8.0
    max_num_nbr=12
    decay_fcn=inverse_square

    # basic setup
    train_frac = 0.8
    #num_epochs = 30
    num_epochs = 10
    num_train = Int32(round(train_frac * num_pts))
    num_test = num_pts - num_train
    # where to find the data
    prop = "formation_energy_per_atom"
    datadir = "/home/rkurchin/CGCNN_formationenergy/data/"
    id = "task_id" # field by which to label each input material
    el_list_dir = string(datadir,prop,"_ellists/")
    graph_weights_dir = string(datadir,prop,"_graphs/")
    decay_fcn_names = Dict(inverse_square=>"invsq", exp_decay=>"exp")
    params_suffix = string('r',cutoff_radius,"_maxnbs",max_num_nbr,"_decay",decay_fcn_names[decay_fcn])
    el_list_subdir = string(el_list_dir, params_suffix,'/')
    gr_wt_subdir = string(graph_weights_dir, params_suffix,'/')

    num_features = sum(num_bins) # we'll use this later
    atom_feature_vecs = make_feature_vectors(features, num_bins, logspaced)

    # dataset...first, read in outputs
    info = CSV.read(string(datadir,prop,".csv"))
    y = Array(Float32.(info[!, Symbol(prop)]))

    # shuffle data and pick out subset
    indices = shuffle(1:size(info,1))[1:num_pts]
    info = info[indices,:]
    output = y[indices]

    element_lists = Array{String}[]
    inputs = FeaturedGraph{SimpleWeightedGraph{Int64, Float32}, Array{Float32,2}}[]

    for r in eachrow(info)
        gr_path = string(gr_wt_subdir, r[Symbol(id)], ".txt")
        els_path = string(el_list_subdir, r[Symbol(id)], ".txt")
        gr = SimpleWeightedGraph(Float32.(readdlm(gr_path)))
        els = readdlm(els_path)
        push!(element_lists, els)
        feature_mat = hcat([atom_feature_vecs[e] for e in els]...)
        input = FeaturedGraph(gr, feature_mat)
        push!(inputs, input)
    end

    # pick out train/test sets
    train_output = output[1:num_train]
    test_output = output[num_train+1:end]
    train_input = inputs[1:num_train]
    test_input = inputs[num_train+1:end]
    train_data = zip(train_input, train_output)

    # build the network
    if pool_type=="mean"
        model = Chain(CGCNConv(num_features=>atom_fea_len), [CGCNConv(atom_fea_len=>atom_fea_len) for i in 1:num_conv-1]..., CGCNMeanPool(crys_fea_len, 0.1), [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers-1]..., Dense(crys_fea_len, 1))
    elseif pool_type=="max"
        model = Chain([CGCNConv(num_features=>num_features) for i in 1:num_conv]..., CGCNMaxPool(crys_fea_len, 0.1), [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers-1]..., Dense(crys_fea_len, 1))
    else
        println("invalid pool type")
    end

    # define loss function
    # TODO: figure out why backprop doesn't work through the regularized version
    #loss(x,y) = Flux.mse(model(x), y) + reg_coeff * sum(norm, params(model))
    loss(x,y) = Flux.mse(model(x), y)
    evalcb() = @show(mean(loss.(test_input, test_output)))
    start_err = evalcb()

    # train
    opt = ADAM(lr)
    _, train_time, mem, _, _ = @timed @epochs num_epochs Flux.train!(loss, params(model), train_data, opt, cb=Flux.throttle(evalcb, 5))

    end_err = evalcb()

    loss_mae(x,y) = Flux.mae(model(x),y)
    end_mae = mean(loss_mae.(test_input, test_output))

    start_err, end_err, end_mae, train_time
end
