#=
Note: MetaGraphs would seem on face to be the elegant way to store atom features, bond lengths, etc., but in practice it's much clunkier because it doesn't support the same functions and GeometricFlux doesn't really play nice with it. So for now this is all done with SimpleWeightedGraphs. Can revisit MetaGraphs in the future if this situation changes...
=#

using PyCall
using GraphPlot, Colors
using LightGraphs, SimpleWeightedGraphs # need LightGraphs for adjacency_matrix fcn

# a few fcns just for readability...

"Return the index of a given site in the structure."
site_index(site) = convert(UInt16, get(site, 2)) + 1

"Return the distance associated with a site in a neighbor list."
site_distance(site) = convert(Float64, get(site, 1))

# these next two functions return the information for the first species in a list – there should only be one because otherwise the structure would be disordered and we probably shouldn't be building a graph...(or maybe we add some fancy functionality later to do superpositions of species?)

"Return atomic number associated with a site."
site_atno(site) = [e.Z for e in site.species.elements][1]

"Return atomic symbol associated with a site."
site_element(site) = [e.symbol for e in site.species.elements][1]

"""
    are_equidistant(site1, site2)

Check if site1 and site2 are equidistant to within tolerance atol, in angstroms (for cutting off neighbor lists consistently).

Note that this only works if site1 and site2 are from a neighbor list from the same central atom.
"""
function are_equidistant(site1, site2, atol=1e-4)
    isapprox(site_distance(site1), site_distance(site2), atol=atol)
end

# options for decay of bond weights with distance...
inverse_square(x) = x^-2.0
exp_decay(x) = exp(-x)

"""
Function to build graph from a CIF file of a crystal structure.

Note that `max_num_nbr` is a "soft" max, in that if there are more of the same distance as the last, all of those will be added (may reconsider this later if it makes things messy)

# Arguments
- `cif_path::String`: path to CIF file
- `radius::Float=8.0`: cutoff radius for atoms to be considered neighbors (in angstroms)
- `max_num_nbr::Integer=12`: maximum number of neighbors to include (even if more fall within cutoff radius)
- `dist_decay_func`: function (e.g. inverse_square or exp_decay) to determine falloff of graph edge weights with neighbor distance

# TODO
- option to cut off by nearest, next-nearest, etc. by DISTANCE rather than NUMBER of neighbors
"""
function build_graph(cif_path; radius=8.0, max_num_nbr=12, dist_decay_func=inverse_square, normalize=true)
    s = pyimport("pymatgen.core.structure")
    c = s.Structure.from_file(cif_path)
    num_atoms = size(c)[1]

    # list of atom symbols
    atom_ids = [site_element(s) for s in c]

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
    weight_mat = zeros(Float32, num_atoms, num_atoms)
    for atom_ind in 1:num_atoms
        this_atom = get(c, atom_ind-1)
        atom_nbs = all_nbrs[atom_ind]
        # iterate over each neighbor...
        for nb_num in 1:size(all_nbrs[atom_ind])[1]
            nb = atom_nbs[nb_num]
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

    # average across diagonal (because neighborness isn't strictly symmetric in the way we're defining it here)
    weight_mat = 0.5.* (weight_mat .+ weight_mat')

    # normalize weights
    if normalize
        weight_mat = weight_mat ./ maximum(weight_mat)
    end

    # turn into a graph...
    g = SimpleWeightedGraph{Int32, Float32}(weight_mat)

    return (g, atom_ids)
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

"Get a list of colors to use for graph visualization."
function graph_colors(atno_list, seed_color=colorant"cyan4")
    atom_types = unique(atno_list)
    atom_type_inds = Dict(atom_types[i]=>i for i in 1:length(atom_types))
    color_inds = [atom_type_inds[i] for i in atno_list]
    colors = distinguishable_colors(length(atom_types), seed_color)
    return colors[color_inds]
end

"Compute edge widths (proportional to weights on graph) for graph visualization."
function graph_edgewidths(g, weight_mat)
    edgewidths = []
    # should be able to do this as
    for e in edges(g)
        append!(edgewidths, weight_mat[e.src, e.dst])
    end
    return edgewidths
end

"Visualize a given graph."
function visualize_graph(g, element_list)
    # gplot doesn't work on weighted graphs
    sg = SimpleGraph(adjacency_matrix(g))
    plt = gplot(sg, nodefillc=graph_colors(element_list), nodelabel=element_list, edgelinewidth=graph_edgewidths(sg, g.weights))
    display(plt)
end
