# Materials Project Database

In this example, we will create and train a neural network on the property `formation_energy_per_atom` based on the architecture as introduced in [this paper](https://arxiv.org/abs/1710.10324), using the [Materials Project](https://materialsproject.org/) database.

## 1. Set up the Dataset

### a. Set up required dependencies

Create an [API key](https://materialsproject.org/open) with [Materials Project](https://materialsproject.org/) to download the training dataset for this example.

Python packages (primarily pymatgen) are also required as dependencies. The easiest way to install these in a new environment, is using [Conda](https://docs.conda.io/en/latest/), by running the following commands.

`conda create -name agn_example1`\
`conda activate agn_example1`\
`conda install -c conda-forge pymatgen=2022.0.4`

*Note:*  Alternatively, to install dependencies in an existing environment, you can skip the first step, activate your environment, and directly install pymatgen=2022.0.4.

### b. Download the Data

To download the structures, simply run

`python download_data.py "YOUR_API_KEY"`\
where, `"YOUR_API_KEY"` is replaced with your actual API key obtained from the Materials Project database.

This downloads the data required to train the models into the directory `example1/data`.

## 2. Train the Network

Now that the dataset has been set up, run the `formation_energy.jl` file in your Julia environment and play around with the training function!

## Note

- By default, the property chosen for training is formation energy per atom. If you want to train a different property, replace it with its pymatgen string in the appropriate line. This change must be reflected in both `download.py` and `example.jl`.

- There may be some CIF parsing warnings from pymatgen that show up, but these shouldn't affect things and can be safely ignored.
