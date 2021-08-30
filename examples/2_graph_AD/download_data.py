from pymatgen.ext.matproj import MPRester
import pandas as pd
import os
from os import path
import sys

id = "task_id"
prop = "final_energy"
dir_path = path.dirname(path.realpath(__file__))
data_dir = dir_path + "/data/"

# create a ./data folder if it doesn't already exist
if not path.isdir(data_dir):
    os.mkdir(data_dir)

cif_folder = data_dir + prop + "_cifs/"
csv_path = data_dir + prop + ".csv"
api_key = sys.argv[1]

m = MPRester(api_key=api_key)

# skip radioactive elements and noble gases
elements = ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"]

# query for property data of stable, ordered materials
data = m.query(criteria={"is_ordered":True, "e_above_hull":0}, properties=["full_formula",id,prop,"cif","elements"])

# convert to a dataframe
df = pd.DataFrame.from_records(data)

# drop out entries that have elements not in the list
inds_to_drop = []
for row in df.iterrows():
    if not set(row[1]["elements"]).intersection(set(elements))==set(row[1]["elements"]):
        inds_to_drop.append(row[0])

df = df.drop(inds_to_drop)

# and save as CSV (without cifs or element lists)
df = df.drop(["cif", "elements"], axis=1)

df.to_csv(csv_path, index=False)

# write cifs
if not path.isdir(cif_folder):
    os.mkdir(cif_folder)

for d in data:
    with open(cif_folder+d["task_id"]+".cif", 'w') as f:
        f.write(d["cif"])
