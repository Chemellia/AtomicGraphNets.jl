# Comparison of AtomicGraphNets.jl and cgcnn.py

AtomicGraphNets.jl provides a model-builder for the typical GCNN architecture called `Xie_model`; in reference to [Tian Xie](http://txie.me/), the original developer of [cgcnn.py](https://github.com/txie-93/cgcnn).\
However, there are some differences between how AtomicGraphNets models and those ones work, particularly with respect to the convolutional operation performed.

The [cgcnn.py](https://github.com/txie-93/cgcnn) package was the first major package to implement atomic graph convolutional networks.\
However, the "convolutional" operation they use, while qualitatively similar, is not convolution by the strict definition involving the graph Laplacian.
In their package, they introduce two such operations.

The first operation is expressed as follows.

```math
\begin{aligned}
v^{(t+1)}_{i} = g [(\sum\limits_{j,k}v^{(t)}_{j} \bigoplus u_{(i,j)_{k}})W^{(t)}_{c} + v^{(t)}_{i}W^{(t)}_{s} + b^{(t)}]
\end{aligned}
```

Here, ``v``, ``u`` represent node features and edge features respectively, and ``i``, ``j``, ``k`` index nodes, neighbors of nodes, and edge multiplicities respectively.
Further, ``\bigoplus`` indicates concatenation, and ``g`` is an activation function.

Note that such an operation, which does not make use of the graph Laplacian, requires explicit computation of neighbor lists for every node, and that the convolutional weight matrix is of very large dimension due to the concatenation step.

The original CGCNN paper explores the following slightly more complicated operation that resulted in better performance.

```math
\begin{aligned}
v^{(t+1)}_i = v^{(t)}_i + \sum\limits_{j,k}\sigma(z^{(t)}_{(i,j)_{k}}W^{(t)}_{f} + b^{(t)}_{f}) \bigodot g(z^{(t)}_{(i,j)_{k}}W^{(t)}_{s} + b^{(t)}_{s}))
\end{aligned}
```

where ``z`` is a concatenation of neighbor features and edge features, and ``\bigodot`` indicates element-wise multiplication.\
This operation entails yet more trainable parameters, and neither operation is particularly performant because the concatenation operation must be done at each step of the forward pass.

The operation implemented in AtomicGraphnets is as follows.

```math
\begin{aligned}
X^{(t+1)} = n_{z}[g[W^{(t)}_{c} \cdot X^{(t)} \cdot L + W^{(t)}_{s} \cdot X^{(t)} + B^{(t)}]]
\end{aligned}
```

where ``X`` is a feature matrix constructed by stacking feature vectors, ``B`` is a bias matrix (stacked identical copies of the per-feature bias vector) and ``n_{z}`` is the z-score normalization (or regularized normalization operation), which we have found to improve stability.

In addition, since the graph Laplacian need only be computed once (and is in fact stored as part of the AtomGraph type), the forward pass is much more computationally efficient.\
Since no concatenation occurs, weight matrices are also smaller, meaning the model has fewer trainable parameters, and no sacrifice in accuracy that we have been able to observe, indicating comparable expressivity.

It is worth noting that one advantage of the [cgcnn.py](https://github.com/txie-93/cgcnn) approach is that it allows for explicitly enumerating edge features.\
In the current version of AtomicGraphNets, the only features of graph edges are their weights.\
Convolutional operations that allow for edge features are under consideration for future versions.
