#=
 Train a simple network to predict formation energy per atom (downloaded from Materials Project).
=#
using CSV, DataFrames
using Random, Statistics
using Flux
using Flux: @epochs
using ChemistryFeaturization
using AtomicGraphNets

function build_graphs(prop, featurization; data_dir = nothing,
			                   num_pts = nothing,
				           verbose = false)
    info = CSV.read(joinpath(data_dir, prop * ".csv"), DataFrame)
    y = Array(Float32.(info[!, Symbol(prop)]))

    # shuffle data and pick out subset
    indices = shuffle(1:size(info, 1))
    if !isnothing(num_pts)
        indices = indices[1:num_pts]
    end
    info = info[indices, :]
    output = y[indices]

    # next, make and featurize graphs
    if verbose
        println("Building graphs and feature vectors from structures...")
    end

    inputs = FeaturizedAtoms[]
    id = "task_id"
    for r in eachrow(info)
        cifpath = string(joinpath(data_dir, prop * "_cifs", r[Symbol(id)] * ".cif"))
        gr = AtomGraph(cifpath)
        if gr === missing
            continue
        end
        input = featurize(gr, featurization)
        push!(inputs, input)
    end
    inputs, output
end

function train_formation_energy(;
    num_pts = 31417, # 100,
    num_epochs = 5,
    data_dir = joinpath(@__DIR__, "data"),
    verbose = true,
)
    println("Setting things up...")

    # data-related options
    train_frac = 0.8 # what fraction for training?
    num_train = Int32(round(train_frac * num_pts))
    num_test = num_pts - num_train
    prop = "formation_energy_per_atom"

    # set up the featurization
    featurization = GraphNodeFeaturization([
        "Group",
        "Row",
        "Block",
        "Atomic mass",
        "Atomic radius",
        "X",
    ])
    num_features =
        sum(ChemistryFeaturization.FeatureDescriptor.output_shape.(featurization.features)) # TODO: update this with cleaner syntax once new version of CF is tagged that has it

    # model hyperparameters – keeping it pretty simple for now
    num_conv = 3 # how many convolutional layers?
    crys_fea_len = 32 # length of crystal feature vector after pooling (keep node dimension constant for now)
    num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
    opt = ADAM(0.001) # optimizer


    inputs, output = build_graphs(prop, featurization; data_dir, num_pts, verbose)

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
