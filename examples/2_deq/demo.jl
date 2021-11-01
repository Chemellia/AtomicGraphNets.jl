using Serialization
using ChemistryFeaturization
using AtomicGraphNets

inputs = deserialize.(readdir("examples/2_deq/data/inputs/", join=true))

l = AGNConvDEQ(61=>61)

l(inputs[1]) # this will just hang...