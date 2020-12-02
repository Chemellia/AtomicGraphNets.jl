# Example 3: Deep Equilibrium Model

**NB:** While this code runs, it is too slow at the moment to be practical (~80-90 min for a single forward pass). Still needs troubleshooting...

The idea here is to implement a [Deep Equilibrium Model](https://arxiv.org/abs/1909.01377) approach with the graph convolution. This particular QM9 task seemed to show better and better performance with adding more convolutional layers, so the idea seemed potentially fruitful here.