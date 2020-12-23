# AtomicGraphNets.jl
![Run tests](https://github.com/aced-differentiate/AtomicGraphNets.jl/workflows/Run%20tests/badge.svg)
[![codecov](https://codecov.io/gh/aced-differentiate/AtomicGraphNets.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aced-differentiate/AtomicGraphNets.jl)

AtomicGraphNets.jl implements graph-based models for machine learning on atomic systems, such as [Crystal Graph Convolutional Neural Nets](https://arxiv.org/abs/1710.10324), in Julia. It makes use of the [Flux](https://fluxml.ai) ecosystem for model building and the [JuliaGraphs](https://github.com/JuliaGraphs) ecosystem for graph representation and visualization, as well as adapting some features from [GeometricFlux](https://github.com/yuehhua/GeometricFlux.jl).


## Getting Started

1. Clone this package to wherever you want to play.

2. Go and try out the example in examples/example1/ – it has its own README file with detailed instructions.


## Future plans
* make docs
* more network architectures (see issues for some ideas)

## Contact
Please feel free to fork and play, and reach out here on GitHub or to rkurchin [at] cmu [dot] edu with suggestions, etc.!

## Acknowledgements
Many thanks to [Dhairya Gandhi](https://github.com/DhairyaLGandhi) for helping out with some adjoints to actually make these layers trainable! :D