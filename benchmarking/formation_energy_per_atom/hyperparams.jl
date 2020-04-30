#=
 Initial hyperparameter optimization for comparison of CrystalGraphConvNets.jl to CGCNN.py.
=#

using Distributed

addprocs(13)

@everywhere include("/home/rkurchin/CGCNN_formationenergy/train_fcn.jl")
using DataFrames

#nums_pts = Int64.(round.(exp10.(range(3,stop=log10(34339),length=6))))
# hyperparameters from Tian's SI, just trying to match that as closely as possible for now
#nums_pts = [Int64(n/0.8) for n in [100, 400, 1600, 6400]]
nums_pts = [6250, 6250]
#nums_pts = Int64.(round.(1.25*exp10.(range(2,stop=log10(32530*0.8),length=8))))
#nums_pts = vcat(nums_pts, nums_pts, nums_pts)
nums_conv = [1,3,5]
#nums_conv = [3]
atom_fea_lens = [10, 20, 50, 100, 200]
#atom_fea_lens = [20, 40, 80, 160]
#atom_fea_lens = [32]
pool_types = ["mean"]
crys_fea_lens = [128]
nums_hidden_layers = [1,2,3,5]
#nums_hidden_layers = [1]
lrs = exp.(range(-8,stop=-3,length=4))
#lrs = exp.(range(-7,stop=-5,length=3))
#lrs = exp10.(range(-3,stop=-1.5,length=4))
#lrs = exp10.(range(-3.2,stop=-2.8,length=5))
#lrs = [best_one]
#features = [["group", "row", "block", "atomic_mass", "atomic_radius", "X"]]
features = [["group", "row", "X", "atomic_radius", "block"]]
nums_bins = [[18, 8, 10, 10, 4]]
logspaceds = [[false, false, false, true, false]]
#reg_coeffs = exp.([-6, -3, 0])

#param_sets = [p for p in Iterators.product(nums_pts, nums_conv, atom_fea_lens, pool_types, crys_fea_lens, nums_hidden_layers, lrs, features, nums_bins, logspaceds, reg_coeffs)]
param_sets = [p for p in Iterators.product(nums_pts, nums_conv, atom_fea_lens, pool_types, crys_fea_lens, nums_hidden_layers, lrs, features, nums_bins, logspaceds)]

results = pmap(cgcnn_train, param_sets)

output = DataFrame(num_conv=Int[], atom_fea_len=Int[], num_hidden_layers=Int[], lr=Float32[], start_err=Float32[], end_err=Float32[], train_time=Float32[])

for i in 1:prod(size(param_sets))
    params = param_sets[i]
    result = results[i]
    row = (params[2], params[3], params[6], params[7], result...)
    push!(output, row)
end

CSV.write("hyperparameter_test_1.csv", output)
