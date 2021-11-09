#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#
using Pkg; Pkg.activate("../../")
using CSV, DataFrames
using Random, Statistics
using Flux
using Flux: @epochs
using ChemistryFeaturization
using AtomicGraphNets
using Serialization

function train(;
    num_pts = 100,
    num_epochs = 5,
    data_dir = joinpath(@__DIR__, "data"),
    verbose = true,
               )
    
    if verbose:
        println("Setting things up...")
    end

    # data-related options
    train_frac = 0.8 # what fraction for training?
    num_train = Int32(round(train_frac * num_pts))
    num_test = num_pts - num_train
    prop = :Cv # choose any column from labels.csv except :key
    id = :key # field by which to label each input material

    # model hyperparameters – keeping it pretty simple for now
    num_conv = 3 # how many convolutional layers?
    crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
    num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
    opt = ADAM(0.001) # optimizer

    # dataset...first, read in outputs
    info = CSV.read(string(data_dir, "labels.csv"), DataFrame)
    y = Array(Float32.(info[!, prop]))

    # shuffle data and pick out subset
    indices = shuffle(1:size(info, 1))[1:num_pts]
    info = info[indices, :]
    output = y[indices]

    # next, read in prefeaturized graphs
    if verbose
        println("Reading in graphs...")
    end
    
    inputs = FeaturizedAtoms[]

    for r in eachrow(info)
        fpath = string(data_dir, "qm9_jls/", r[id], ".jls")
        push!(inputs, deserialize(fpath))
    end

    # pick out train/test sets
    if verbose
        println("Dividing into train/test sets...")
    end
    train_output = output[1:num_train]
    test_output = output[num_train+1:end]
    train_input = inputs[1:num_train]
    test_input = inputs[num_train+1:end]
    train_data = zip(train_input, train_output)

    # build the model
    if verbose
        println("Building the network...")
    end
    num_features = size(inputs[1].encoded_features, 1)
    model = build_CGCNN(
        num_features,
        num_conv = num_conv,
        atom_conv_feature_length = crys_fea_len,
        pooled_feature_length = (Int(crys_fea_len / 2)),
        num_hidden_layers = 1,
    )

    # define loss function and a callback to monitor progress
    loss(x, y) = Flux.Losses.mse(model(x), y)
    evalcb_verbose() = @show(mean(loss.(test_input, test_output)))
    evalcb_quiet() = return nothing
    evalcb = verbose ? evalcb_verbose : evalcb_quiet
    evalcb()

    # train
    if verbose
        println("Training!")
    end
    @epochs num_epochs Flux.train!(
        loss,
        params(model),
        train_data,
        opt,
        cb = Flux.throttle(evalcb, 5),
    )

    return model
end
