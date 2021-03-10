# QM9 Dataset

In this example, we will create and train a neural network based on the architecture as introduced in [this paper](https://arxiv.org/abs/1710.10324), using the [QM9](http://quantum-machine.org/datasets/) dataset.

## Train the Network

Run the `qm9.jl` script in your Julia environment and see what happens!

*Note:* The `.xyz` files provided within the QM9 dataset are not parsable directly by ASE. So, the last couple lines need to be removed, which is easy enough to be done using a simple script. For convenience, and demonstration purposes, a small set of the modified `.xyz` files have been made available here.

## Remarks

It is well worth noting that the actual model performance on QM9 is not that great since we're currently not encoding a variety of important features for organic molecules.\
This is provided mainly to show the processing of a different dataset and demonstrate batch processing capabilities.
