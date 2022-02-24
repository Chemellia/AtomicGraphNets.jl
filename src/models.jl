using Flux: glorot_uniform

"""
Build a model of the architecture introduced in the Xie and Grossman 2018 paper: https://arxiv.org/abs/1710.10324

Input to the resulting model is a FeaturedGraph with feature matrix with `input_feature_length` rows and one column for each node in the input graph.

Network has convolution layers, then pooling to some fixed length, followed by Dense layers leading to output.

# Arguments
- `input_feature_length::Integer`: length of feature vector at each node
- `num_conv::Integer`: number of convolutional layers
- `conv_activation::F`: activation function on convolutional layers
- `atom_conv_feature_length::Integer`: length of output of conv layers
- `pool_type::String`: type of pooling after convolution (mean or max)
- `pool_width::Float`: fraction of atom_conv_feature_length that pooling window should span
- `pooled_feature_length::Integer`: feature length to pool down to
- `num_hidden_layers::Integer`: how many Dense layers before output? Note that if this is set to 1 there will be no nonlinearity imposed on these layers
- `hidden_layer_activation::F`: activation function on hidden layers
- `output_layer_activation::F`: activation function on output layer; should generally be identity for regression and something that normalizes appropriately (e.g. softmax) for classification
- `output_length::Integer`: length of output vector
- `initW::F`: function to use to initialize weights in trainable layers
"""
function build_CGCNN(
    input_feature_length;
    num_conv = 2,
    conv_activation = softplus,
    atom_conv_feature_length = 80,
    pool_type = "mean",
    pool_width = 0.1,
    pooled_feature_length = 40,
    num_hidden_layers = 1,
    hidden_layer_activation = softplus,
    output_layer_activation = identity,
    output_length = 1,
    initW = glorot_uniform,
)
    @assert atom_conv_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    model = Chain(
        AGNConv(
            input_feature_length => atom_conv_feature_length,
            conv_activation,
            initW = initW,
        ),
        [
            AGNConv(
                atom_conv_feature_length => atom_conv_feature_length,
                conv_activation,
                initW = initW,
            ) for i = 1:num_conv-1
        ]...,
        AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width),
        [
            Dense(
                pooled_feature_length,
                pooled_feature_length,
                hidden_layer_activation,
                init = initW,
            ) for i = 1:num_hidden_layers-1
        ]...,
        Dense(pooled_feature_length, output_length, output_layer_activation, init = initW),
    )
end

# these dispatches can be removed once https://github.com/FluxML/Flux.jl/issues/1673 is fixed...
# (m::Parallel)(x) = mapreduce(f -> f(x), m.connection, m.layers)
# (m::Parallel)(xs...) = mapreduce((f, x) -> f(x), m.connection, m.layers, xs)
"""
This is a helper function to the main model builder below. Takes in the inputs and models for the two "parallel" CGCNN-like models at the start of the SGCNN architecture and outputs the concatenated final result.
"""
function slab_graph_layer(
    bulk_graph::FeaturizedAtoms{AtomGraph,GraphNodeFeaturization},
    bulk_model,
    surface_graph::FeaturizedAtoms{AtomGraph,GraphNodeFeaturization},
    surface_model,
)
    bulk_output = bulk_model(bulk_graph)
    surface_output = surface_model(surface_graph)
    vcat(bulk_output, surface_output)
end

"""
Build a slab graph network, based off of Kim et al. 2020: https://pubs.acs.org/doi/10.1021/acs.chemmater.9b03686

For now, fixing both "parallel" convolutional paths to have same hyperparams for simplicity and to basically be copies of CGCNN. Might relax this later.

# Arguments:
Same as [`build_CGCNN`](@ref) except for additional parameter of `hidden_layer_width`
"""
function build_SGCNN(
    input_feature_length::Integer;
    num_conv = 2,
    conv_activation = softplus,
    atom_conv_feature_length = 80,
    pool_type = "mean",
    pool_width = 0.1,
    pooled_feature_length = 40,
    hidden_layer_width = 40,
    num_hidden_layers = 3,
    hidden_layer_activation = softplus,
    output_layer_activation = identity,
    output_length = 1,
    initW = glorot_uniform,
)
    bulk_model = Chain(
        AGNConv(
            input_feature_length => atom_conv_feature_length,
            conv_activation,
            initW = initW,
        ),
        [
            AGNConv(
                atom_conv_feature_length => atom_conv_feature_length,
                conv_activation,
                initW = initW,
            ) for i = 1:num_conv-1
        ]...,
        AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width),
    )
    surface_model = Chain(
        AGNConv(
            input_feature_length => atom_conv_feature_length,
            conv_activation,
            initW = initW,
        ),
        [
            AGNConv(
                atom_conv_feature_length => atom_conv_feature_length,
                conv_activation,
                initW = initW,
            ) for i = 1:num_conv-1
        ]...,
        AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width),
    )
    model = Chain(
        Parallel(vcat, surface_model, bulk_model),
        Dense(
            2 * pooled_feature_length,
            hidden_layer_width,
            hidden_layer_activation,
            init = initW,
        ),
        [
            Dense(
                hidden_layer_width,
                hidden_layer_width,
                hidden_layer_activation,
                init = initW,
            ) for i = 1:num_hidden_layers-1
        ]...,
        Dense(hidden_layer_width, output_length, output_layer_activation, init = initW),
    )
end

"""
Similar to [`build_CGCNN`](@ref) but in the style of Deep Equilibrium Models: https://arxiv.org/abs/1909.01377

Can be thought of as equivalent to a CGCNN with infinite convolutional layers.

Input to the resulting model is a FeaturedGraph with feature matrix with `input_feature_length` rows and one column for each node in the input graph.

# Arguments
- `input_feature_length::Integer`: length of feature vector at each node
- `conv_activation::F`: activation function on convolutional layers
- `pool_type::String`: type of pooling after convolution (mean or max)
- `pool_width::Float`: fraction of atom_conv_feature_length that pooling window should span
- `pooled_feature_length::Integer`: feature length to pool down to
- `num_hidden_layers::Integer`: how many Dense layers before output? Note that if this is set to 1 there will be no nonlinearity imposed on these layers
- `hidden_layer_activation::F`: activation function on hidden layers
- `output_layer_activation::F`: activation function on output layer; should generally be identity for regression and something that normalizes appropriately (e.g. softmax) for classification
- `output_length::Integer`: length of output vector
- `initW::F`: function to use to initialize weights in trainable layers
"""
function build_CGCNN_DEQ(
    input_feature_length;
    conv_activation = softplus,
    pool_type = "mean",
    pool_width = 0.1,
    pooled_feature_length = 40,
    num_hidden_layers = 1,
    hidden_layer_activation = softplus,
    output_layer_activation = identity,
    output_length = 1,
    initW = glorot_uniform,
)
    @assert input_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    model = Chain(
        AGNConvDEQ(input_feature_length, conv_activation, initW = initW),
        AGNPool(pool_type, input_feature_length, pooled_feature_length, pool_width),
        [
            Dense(
                pooled_feature_length,
                pooled_feature_length,
                hidden_layer_activation,
                init = initW,
            ) for i = 1:num_hidden_layers-1
        ]...,
        Dense(pooled_feature_length, output_length, output_layer_activation, init = initW),
    )
end
