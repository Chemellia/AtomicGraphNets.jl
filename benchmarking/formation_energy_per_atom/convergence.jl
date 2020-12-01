#=
 Convergence checks.
=#

using Distributed

addprocs(parse(Int, ENV["SLURM_NTASKS"]))

@everywhere include("/home/rkurchin/CGCNN_formationenergy/train_fcn.jl")
using DataFrames

# hyperparameters from Tian's SI, just trying to match that as closely as possible for now
nums_pts = Int64.(round.(1.25*exp10.(range(2,stop=log10(32530*0.8),length=8))))[7:8]
nums_pts = sort(vcat(nums_pts, nums_pts, nums_pts))
nums_conv = [3]
atom_fea_lens = [32]
pool_types = ["mean"]
crys_fea_lens = [128]
nums_hidden_layers = [1]
#lrs = exp10.(range(-3.2,stop=-2.8,length=5))
lrs = [0.001]
features = [["group", "row", "X", "atomic_radius", "block"]]
nums_bins = [[18, 8, 10, 10, 4]]
logspaceds = [[false, false, false, true, false]]

param_sets = [p for p in Iterators.product(nums_pts, nums_conv, atom_fea_lens, pool_types, crys_fea_lens, nums_hidden_layers, lrs, features, nums_bins, logspaceds)]

results = pmap(cgcnn_train, param_sets)

output = DataFrame(num_pts=Int[], start_err=Float32[], end_err=Float32[], mae=Float32[], train_time=Float32[])

for i in 1:prod(size(param_sets))
    params = param_sets[i]
    result = results[i]
    row = (params[1], result...)
    push!(output, row)
end

CSV.write("convergence_test_highNredux3_10epochs.csv", output)

