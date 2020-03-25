using PyCall
# need pymatgen installed in Julia's pythondir, can do with Conda.jl
using GeometricFlux
using SimpleWeightedGraphs, MetaGraphs
using GraphPlot, Colors

include("functions.jl")

# pick a CIF
cif_dir = "cgcnn/cif_files/"
cif_list = readdir(cif_dir)
# pick a random cif
cif_path = string(cif_dir, cif_list[rand(1:size(cif_list)[1])])
# specific ones...
#cif_path = string(cif_dir, cif_list[1])
#cif_path = "cgcnn/cif_files/VBrI-MoSSe-FM.cif"
print(cif_path)

# read in the CIF and get some basic info
c = s.Structure.from_file(cif_path)
num_atoms = size(c)[1]
# for pulling atom features later...
atno_list = [site_atno(site) for site in c]
element_list = [site_element(site) for site in c]

# do some magic
g = build_graph(cif_path)
visualize_graph(g, element_list)

# make a simple graph convolution layer
l = GCNConv(g, 1=>1)


# checking if neighbor lists make sense...
# i.e., if A is B's neighbor, is B always A's?
#all_nbrs = c.get_all_neighbors(radius, include_index=true)
#nb_indices = [[site_index(site) for site in nb_list] for nb_list in all_nbrs]
# this should return a vector of all the same number...but it doesn't always
#sanity_check = [sum([count(i->(i==k), l) for l in nb_indices]) for k in 1:size(nb_indices,1)]
