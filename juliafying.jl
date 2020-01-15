using PyCall
using GeometricFlux
using LightGraphs, SimpleWeightedGraphs, MetaGraphs
using GraphPlot, Colors
# need pymatgen installed in Julia's pythondir, can do with Conda.jl
include("functions.jl")

# read in a CIF
cifdir = "cgcnn/cif_files/"
cifs = readdir(cifdir)

cif = string(cifdir, cifs[rand(1:size(cifs)[1])]) # random one
#cif = string(cifdir, cifs[1]) # specific one
print(cif)

weight_mat, dist_mat, atno_list, element_list = build_graph_matrices(cif)
num_atoms = size(weight_mat)[1]

# turn into a graph...trying a few options here
g = SimpleGraph(num_atoms)
wg = SimpleWeightedGraph{UInt16, UInt8}(num_atoms)

for i=1:num_atoms, j=1:i
    if weight_mat[i,j] > 0
        add_edge!(g, i, j)
        add_edge!(wg, i, j, weight_mat[i,j])
    end
end

# visualize it...
plt = gplot(g, nodefillc=graph_colors(atno_list), nodelabel=element_list, edgelinewidth=graph_edgewidths(g, weight_mat))
display(plt)

# make a simple graph convolution layer
l = GCNConv(wg, 1=>1)


#=
# make the MetaGraph to store the features
mg = MetaGraph{UInt16, UInt8}(g, 0)

# set vertex features
for v in 1:num_atoms
    set_prop!(mg, v, :atno, atno_list[v])
end

# set edge features
for i=1:num_atoms, j=1:i
    set_prop!(mg, i, j, :len, dist_mat[i,j]) # bond length
    set_prop!(mg, i, j, :weight, weight_mat[i,j]) # number of bonds
end
=#
