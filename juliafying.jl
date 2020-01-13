using PyCall
using GeometricFlux
using LightGraphs, SimpleWeightedGraphs, MetaGraphs
# need pymatgen installed in Julia's pythondir, can do with Conda.jl
include("functions.jl")

# import pymatgen stuff (to read in CIF and find neighbors)
s = pyimport("pymatgen.core.structure")

# read in a CIF
c = s.Structure.from_file("cgcnn/cif_files/Ag2Br2-CH-NM.cif")
num_atoms = size(c)[1]
# for pulling atom features later...
atno_list = [site_atno(site) for site in c]

# set some defaults that we can play with later
# note that `max_num_nbr` is a "soft" max, in that if there are more of the same distance as the twelfth, all of those will be added (may reconsider this later if it makes things messy)
max_num_nbr = 12
radius = 8

# find neighbors, requires a cutoff radius
# returns a NxM Array of PyObject PeriodicSite
# each PeriodicSite is (site, distance, index, image)
# N is num sites in crystal, M is num neighbors
# N=4, M=43 for Ag2Br2-CH-NM with cutoff radius
all_nbrs = c.get_all_neighbors(radius, include_index=true)

# sort by distance
# returns list of length N of lists of length M
all_nbrs = [sort(all_nbrs[i,:], lt=(x,y)->isless(site_distance(x), site_distance(y))) for i in 1:num_atoms]

# iterate through each list of neighbors (corresponding to neighbors of a given atom) to add graph edges
# also store some basic features so we don't have to iterate through all over again when it gets converted to a MetaGraph
dist_mat = zeros(num_atoms, num_atoms)
weight_mat = zeros(UInt8, num_atoms, num_atoms)
for atom_ind in 1:num_atoms
    this_atom = get(c, atom_ind-1)
    atom_nbs = all_nbrs[atom_ind]
    # iterate over each neighbor...
    for nb_num in 1:size(all_nbrs[atom_ind])[1]
        print(atom_ind, ' ', nb_ind, ' ')
        nb = all_nbrs[atom_ind][nb_num]
        println(nb)
        global nb_ind = site_index(nb)
        # if we're under the max, add it for sure
        if nb_num < max_num_nbr
            #add_bond!(g, atom_ind, nb_ind)
            weight_mat[atom_ind, nb_ind] = weight_mat[atom_ind, nb_ind] + 1
            dist_mat[atom_ind, nb_ind] = site_distance(nb)
        # if we're at/above the max, add if distance is the same
        else
            # check we're not on the last one
            if nb_ind < size(atom_nbs)[1] - 1
                next_nb = atom_nbs[nb_ind + 1]
                # add another bond if it's the exact same distance to the next neighbor in the list
                if are_equidistant(nb, next_nb)
                    weight_mat[atom_ind, nb_ind] = weight_mat[atom_ind, nb_ind] + 1
                end
            end
        end
    end
end

# turn into a graph
g = SimpleGraph(num_atoms)
for i in 1:num_atoms
    for j in 1:i
        if weight_mat[i,j] > 0
            add_edge!(g, i, j)
        end
    end
end

# make the MetaGraph to store the features
mg = MetaGraph{UInt16, UInt8}(g, 0)

# set vertex features
#for v in 1:num_atoms
#    set_prop!(mg, v, atno_list[v])
#end

# set edge features
#for i in 1:num_atoms
#    for j in 1:i
#        set_prop!(mg, i, j, dist_mat[i,j])
#    end
#end
