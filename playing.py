from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

from random import sample

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import Normalizer

# this should work (provided the dirs Loss_per_Epoch, Checkpoints, Models, and Test_results exist)
# python main.py --csv_dir ./csv_files/ --cif_dir ./cif_files/ --init_dir ./atom_init/ --prop 'vbm' --pool_func 'mean'

# now let's take it all apart a little bit
# valid properties to use are titles of csv_files
dataset = CIFData('./csv_files/', './cif_files/', './atom_init/', 'vbm')

example = dataset[0]
name = example[2]
prop = example[1]

atom_fea = example[0][0] # shape is 4 x 92 (num_atoms x length of atom feature vector)
nbr_fea = example[0][1] # 4 x 12 x 41 (Gaussian filtered neighbor distances where last dimension is location of the mean from min_radius to cutoff)
nbr_fea_idx = example[0][2] # 4 x 12 (index of each neighbor type...this should be convertible to adjacency matrix with a bit of thinking)

# this next bit from the ConvLayer stuff
atom_nbr_fea = atom_fea[nbr_fea_idx, :] # this is 4 x 12 x 92, so it expands the first dim to a matrix because nbr_fea_idx is one and then does this at every feature along the second dimension (now third)
N, M = nbr_fea_idx.shape
atom_fea_len = 92
nbr_fea_len = 41
total_nbr_fea = torch.cat([atom_fea.unsqueeze(1).expand(N, M, atom_fea_len), atom_nbr_fea, nbr_fea], dim=2) # concatenate all the features together (unsqueeze adds singleton dimension, expand replicates to the prescribed dimensions (in this case along M, dimension 1)), final shape is 4 x 12 x 225
fc_full = nn.Linear(2*atom_fea_len+nbr_fea_len, 2*atom_fea_len)
total_gated_fea = fc_full(total_nbr_fea) # 4 x 12 x 184

# now compare this directly to main.py (from line 135)
collate_fn = collate_pool

train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset=dataset,
    collate_fn=collate_fn,
    batch_size=256,
    train_ratio=0.8,
    num_workers=1,
    val_ratio=0.1,
    test_ratio=0.1,
    pin_memory=False,
    train_size=None,
    val_size=None,
    test_size=None,
    return_test=True)

sample_data_list = [dataset[i] for i in
                    sample(range(len(dataset)), 500)]
_, sample_target, _ = collate_pool(sample_data_list)
normalizer = Normalizer(sample_target)

structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3,
h_fea_len=128, n_h=1, pool_func='mean',
lassification=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.01, weight_decay=0)
