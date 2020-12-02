using Flux
include("layers.jl")

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
function Xie_model(input_feature_length; num_conv=2, conv_activation=softplus, atom_conv_feature_length=80, pool_type="mean", pool_width=0.1, pooled_feature_length=40, num_hidden_layers=1, hidden_layer_activation=softplus, output_layer_activation=identity, output_length=1, initW=glorot_uniform)
    # add check that each feature length gets shorter, throw warning if not
    model = Chain(AGNConv(input_feature_length=>atom_conv_feature_length, conv_activation, initW=initW), [AGNConv(atom_conv_feature_length=>atom_conv_feature_length, conv_activation, initW=initW) for i in 1:num_conv-1]..., AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width), [Dense(pooled_feature_length, pooled_feature_length, hidden_layer_activation, initW=initW) for i in 1:num_hidden_layers-1]..., Dense(pooled_feature_length, output_length, output_layer_activation, initW=initW))
end
