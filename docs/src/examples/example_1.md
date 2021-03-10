# Materials Project Database

In this example, we will create and train a neural network based on the architecture as introduced in [this paper](https://arxiv.org/abs/1710.10324), using the [Materials Project](https://materialsproject.org/) database.

## 1. Set up the Dataset

### a. Set up required dependencies

Create an [API key](https://materialsproject.org/open) with [Materials Project](https://materialsproject.org/) to download the training dataset for this example.

Few python packages (like pymatgen) are also required as dependencies. The easiest way to install these in a new environment, is using [Conda](https://docs.conda.io/en/latest/), by running the following commands.

`conda create --name example1 --file conda_env_specfile.txt`\
`conda activate example1`

*Note:*  Alternatively, to install dependencies in an existing environment, replace `create` with `install` and skip running the second command.

### b. Download the Data

To download the structures, simply run

`python download_data.py "YOUR_API_KEY"`\
where, `"YOUR_API_KEY"` is replaced with your actual API key obtained from the Materials Project database.

*Note:*  By default, the property chosen for training is formation energy per atom. If you want to train a different property, replace it with its pymatgen string in the appropriate line. This change must be reflected in both `download.py` and `example.jl`.

### c. Update paths

In `example.jl`, update the `datadir` specified in to point to the directory to which the CIFs were downloaded.

## 2. Train the Network

Now that the dataset has been set up, run the `example.jl` file in your Julia environment and see what happens!

*Note:* There may be some CIF parsing warnings from pymatgen that show up, but these shouldn't affect things and can be safely ignored.

Feel free to peruse and play with other options defined at the top of the file as well and see how it impacts the results!\
In particular, the defaults have been set to only have a dataset of size 100 so that the base case will run quickly, but feel free to try more data to see how much better the results get.
