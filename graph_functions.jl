using PyCall
using GraphPlot, Colors
using LightGraphs, SimpleWeightedGraphs

# import pymatgen stuff (to read in CIF and find neighbors)
global s = pyimport("pymatgen.core.structure")

# a few fcns just for readability
site_index(site) = convert(UInt16, get(site, 2)) + 1
site_distance(site) = convert(Float64, get(site, 1))
site_atno(site) = [e.Z for e in site.species.elements][1] # for now just returns first one, can add checks to handle disordered stuff later maybe
site_element(site) = [e.symbol for e in site.species.elements][1]

#=
check if two sites are equidistant (for cutting off neighbor lists consistently)
tolerance is in angstroms
note that this doesn't check that they're from the same central atom...
=#
function are_equidistant(site1, site2, atol=1e-4)
    isapprox(site_distance(site1), site_distance(site2), atol=atol)
end

# options for decay of bond weights with distance...
inverse_square(x) = x^-2.0
exp_decay(x) = exp(-x)

#=
Function to actually build graph from a CIF file of a crystal structure.
Note that `max_num_nbr` is a "soft" max, in that if there are more of the same distance as the twelfth, all of those will be added (may reconsider this later if it makes things messy)
=#
function build_graph(cif_path; radius=8.0, max_num_nbr=12, dist_decay_func=inverse_square)
    c = s.Structure.from_file(cif_path)
    num_atoms = size(c)[1]
    #= find neighbors, requires a cutoff radius
    returns a NxM Array of PyObject PeriodicSite
    ... except when it returns a list of N of lists of length M...
    each PeriodicSite is (site, distance, index, image)
    N is num sites in crystal, M is num neighbors
    N=4, M=43 for Ag2Br2-CH-NM with cutoff radius
    =#
    all_nbrs = c.get_all_neighbors(radius, include_index=true)

    # sort by distance
    # returns list of length N of lists of length M
    if length(size(all_nbrs)) == 2
        all_nbrs = [sort(all_nbrs[i,:], lt=(x,y)->isless(site_distance(x), site_distance(y))) for i in 1:num_atoms]
    elseif length(size(all_nbrs)) == 1
        all_nbrs = [sort(all_nbrs[i][:], lt=(x,y)->isless(site_distance(x), site_distance(y))) for i in 1:num_atoms]
    end

    # iterate through each list of neighbors (corresponding to neighbors of a given atom) to find bonds (eventually, graph edges)
    weight_mat = zeros(num_atoms, num_atoms)
    for atom_ind in 1:num_atoms
        this_atom = get(c, atom_ind-1)
        atom_nbs = all_nbrs[atom_ind]
        # iterate over each neighbor...
        for nb_num in 1:size(all_nbrs[atom_ind])[1]
            nb = all_nbrs[atom_ind][nb_num]
            global nb_ind = site_index(nb)
            # if we're under the max, add it for sure
            if nb_num < max_num_nbr
                weight_mat[atom_ind, nb_ind] = weight_mat[atom_ind, nb_ind] + dist_decay_func(site_distance(nb))
            # if we're at/above the max, add if distance is the same
            else
                # check we're not on the last one
                if nb_ind < size(atom_nbs)[1] - 1
                    next_nb = atom_nbs[nb_ind + 1]
                    # add another bond if it's the exact same distance to the next neighbor in the list
                    if are_equidistant(nb, next_nb)
                        weight_mat[atom_ind, nb_ind] = weight_mat[atom_ind, nb_ind] + dist_decay_func(site_distance(nb))
                    end
                end
            end
        end
    end

    # normalize weights
    weight_mat = weight_mat ./ maximum(weight_mat)

    # average across diagonal (because neighborness isn't strictly symmetric in the way we're defining it here)
    weight_mat = 0.5.* (weight_mat .+ weight_mat')

    # turn into a graph...
    g = SimpleWeightedGraph{UInt16, Float32}(num_atoms)

    for i=1:num_atoms, j=1:i
        if weight_mat[i,j] > 0
            add_edge!(g, i, j, weight_mat[i,j])
        end
    end

    return g
end

# function to check if two sites are the same
#=
function are_same(site1, site2, atol=0.1)
    #site1.is_periodic_image(site2) might also work
    site1.distance(site2) < atol # angstroms
end
=#

# function to add a bond in graph
# either increment weight by 1 or create the weight
#function add_bond!(g, ind1, ind2)
#    curr_wt = g.weights[ind1, ind2]
#    add_edge!(g, ind1, ind2, curr_wt + 1)
#end

# to make node colors for graph visualization
function graph_colors(atno_list, seed_color=colorant"cyan4")
    atom_types = unique(atno_list)
    atom_type_inds = Dict(atom_types[i]=>i for i in 1:size(atom_types)[1])
    color_inds = [atom_type_inds[i] for i in atno_list]
    colors = distinguishable_colors(size(atom_types)[1], seed_color)
    return colors[color_inds]
end

# return vector of edge widths proportional to number of bonds
function graph_edgewidths(g, weight_mat)
    edgewidths = []
    # should be able to do this as
    for e in edges(g)
        append!(edgewidths, weight_mat[e.src, e.dst])
    end
    return edgewidths
end

function visualize_graph(g, element_list)
    # gplot doesn't work on weighted graphs
    sg = SimpleGraph(adjacency_matrix(g))
    plt = gplot(sg, nodefillc=graph_colors(element_list), nodelabel=element_list, edgelinewidth=graph_edgewidths(sg, g.weights))
    display(plt)
end
