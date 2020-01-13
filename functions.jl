# a few fcns just for readability
site_index(site) = convert(UInt16, get(site, 2)) + 1
site_distance(site) = convert(Float64, get(site, 1))
site_atno(site) = [e.Z for e in site.species.elements][1] # for now just returns first one, can add checks to handle disordered stuff later maybe

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
