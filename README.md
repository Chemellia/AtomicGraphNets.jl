# AtomicGraphNets.jl
![Run tests](https://github.com/aced-differentiate/AtomicGraphNets.jl/workflows/Run%20tests/badge.svg)
[![codecov](https://codecov.io/gh/aced-differentiate/AtomicGraphNets.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aced-differentiate/AtomicGraphNets.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chemellia.github.io/AtomicGraphNets.jl/dev/)

AtomicGraphNets.jl implements graph-based models for machine learning on atomic systems, such as [Crystal Graph Convolutional Neural Nets](https://arxiv.org/abs/1710.10324), in Julia. It makes use of the [Flux](https://fluxml.ai) ecosystem for model building and the [JuliaGraphs](https://github.com/JuliaGraphs) ecosystem for graph representation and visualization, as well as adapting some features from [GeometricFlux](https://github.com/yuehhua/GeometricFlux.jl).

Documentation is in progress [over here](https://chemellia.github.io/AtomicGraphNets.jl/dev/).

## Getting Started

1. To install the latest tagged version, in your Julia REPL, do `]add AtomicGraphNets`. However, you can also play with the latest version on the `main` branch by skipping to step 2 and then doing `]add /path/to/repo` where you replace the dummy path with the location of your clone.

2. Clone this package to wherever you want to play.

3. Go and try out the example in examples/example1/ – it has its own README file with detailed instructions.

## Contributing
We welcome community contributions! Please refer to [contribution guide](CONTRIBUTING.md) for suggestions on how to go about things.

## Acknowledgements
Many thanks to [Dhairya Gandhi](https://github.com/DhairyaLGandhi) for helping out with some adjoints to actually make these layers trainable! :D
