# Graph Convolutional Neural Networks

## How does AI learn structures?

GCNNs have proven to be one of the best ways using which AI can learn structures represented as graphs.

## Why Graphs?

Graphs provide us with a way to mathematically and explicitly represent complex information; and that includes atomic structure of molecules as well.

## Why CNNs?

* CNNs have fixed parameters. This allows us to observe a low memory footprint and lesser computational cost, relative to other deep learning methods.
* CNNs use a local kernel, which lets us build heirarchies of information as per requirement.
* CNNs allow us to retain spatial invariance properties of our data. This is especially beneficial while dealing with information that can be represented as geometric structures.

## Why Graphs + CNNs?

Machine Learning today is typically constrained to dealing with data represented using what can be understood to be "1-dimensional, regular, and uniform data structures"; and identifying some properties (typically euclidean-based) represented that the machine learning algorithm can exploit.\
However, if presented with data represented in a structure (such as graphs) that doesn't quite fit this category, then typically the first step followed is to steamroll the data (using techniques such as dimensionality reduction) in an attempt to flatten it out, and essentially vectorize it.\
However, in doing so, vital spatial aspects of the data, and structural information related to problem gets discarded.\
This can lead to wrong relational information being learnt by the model; and can also sometimes principally prove to be counterproductive.

For this reason, we try and employ Graph Convolutional Neural Networks.\
Graphs are an excellent way for representing relationships and models (such as a molecule).\
With graphs, we try to ensure that along with the node's properties, aggregated information regarding the node's neighbourhood is also represented. When used in conjunction with CNNs we are presented with a way to combine this information (using some mathematical operations and functions), and to convert it into a higher level representation that can be more useful to the model.\
