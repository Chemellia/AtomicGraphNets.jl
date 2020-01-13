using PyCall
using GeometricFlux
using LightGraphs, SimpleWeightedGraphs, MetaGraphs
# need pymatgen installed in Julia's pythondir, can do with Conda.jl

# import the code
s = pyimport("pymatgen.core.structure")

# read in a CIF
c = s.Structure.from_file("cgcnn/cif_files/Ag2Br2-CH-NM.cif")

# set some defaults that we can play with later
# this is a "soft" max, in that if there are more of the same distance as the twelfth, all of those will be added (may reconsider this later if it makes things messy)
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
all_nbrs = [sort(all_nbrs[i,:], lt=(x,y)->isless(site_distance(x), site_distance(y))) for i in 1:size(all_nbrs)[1]]
num_atoms = size(all_nbrs)[1]

# build graph with a vertex for each atom
# should eventually use MetaGraph to do features
g = SimpleWeightedGraph{UInt16, UInt8}(num_atoms)

# iterate through each list of neighbors (corresponding to neighbors of a given atom) to add graph edges
# also store some basic features so we don't have to iterate through all over again when it gets converted to a MetaGraph
dist_mat = zeros(num_atoms, num_atoms)
atom_ind = 1
for atom_nbs in all_nbrs
    this_atom = get(c, atom_ind-1)
    # iterate over each neighbor...
    global nb_ind = 1
    for nb in atom_nbs
        println(atom_ind, nb_ind)
        # if we're under the max, add it for sure
        if nb_ind < max_num_nbr
            add_bond!(g, atom_ind, site_index(nb))
        # if we're at/above the max, add if distance is the same
        else
            # check we're not on the last one
            if nb_ind < size(atom_nbs)[1] - 1
                next_nb = atom_nbs[nb_ind + 1]
                # add another bond if it's the exact same distance to the next neighbor in the list
                if are_equidistant(nb, next_nb)
                    add_bond!(g, atom_ind, site_index(nb))
                end
            end
        end
        global nb_ind = nb_ind + 1
    end
    global atom_ind = atom_ind + 1
end
