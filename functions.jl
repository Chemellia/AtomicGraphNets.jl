# a few fcns just for readability
site_index(site) = convert(UInt16, get(site, 2)) + 1
site_distance(site) = convert(Float64, get(site, 1))
site_atno(site) = [e.Z for e in site.species.elements][1] # for now just returns first one, can add checks to handle disordered stuff later maybe
site_element(site) = [e.symbol for e in site.species.elements][1]

# function to check if two sites are the same
function are_same(site1, site2, atol=0.1)
    #site1.is_periodic_image(site2) might also work
    site1.distance(site2) < atol # angstroms
end

# check if two sites are equidistant (for cutting off neighbor lists consistently)
# tolerance is in angstroms
# note that this doesn't check that they're from the same central atom...
function are_equidistant(site1, site2, atol=1e-4)
    isapprox(site_distance(site1), site_distance(site2), atol=atol)
end

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
