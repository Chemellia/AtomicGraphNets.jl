# this stuff should get moved to ChemistryFeaturization eventually

using Xtals

function xyz_to_crystal(fpath)
    a = read_xyz(fpath)

    # get coord bounds to just make a big box
    coord_max = maximum(a.coords.x, dims=2)
    coord_min = minimum(a.coords.x, dims=2)
    coord_range = coord_max .- coord_min

    box = Box((max.(3 .* coord_range, 16))...) # just make it way bigger than the molecule and at least twice the default cutoff
    charges = Charges(zeros(a.n), a.coords)

    Crystal(splitpath(fpath)[end][1:end-4], box, Frac(a, box), Frac(charges, box))
end

# this stuff is on my local machine...
qm9xyz = readdir("/Volumes/Data/ML_datasets/qm9/", join=true)
xyzs = qm9xyz[rand(1:133886, 100)]

# make the inputs from the filepaths
ags = AtomGraph.(xyz_to_crystal.(xyzs))
fzn = GraphNodeFeaturization([
    "Group",
    "Row",
    "Block",
    "Atomic mass",
    "Atomic radius",
    "X",
])
fas = [featurize(ag, fzn) for ag in ags]