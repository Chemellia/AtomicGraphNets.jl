# Examples

This directory has a few examples which use the same dataset of formation energy per atom from Materials Project. This README gives instructions for getting your environment set up to run the examples.

## 1. Create the dataset

### a) Set up some dependencies...
You'll need to [create an API key](https://materialsproject.org/open) with Materials Project to download the training dataset, so go and do that first.

You'll also need `pymatgen` and a few other Python packages installed. The easiest way to do this is using [Conda](https://docs.conda.io/en/latest/), which can be accomplished as follows.

In this example's directory, run:
```
conda create --name example1 conda_env_specfile.txt
conda activate example1
```
Alternatively, to install dependencies in an existing environment, replace `create` above with `install` and omit the second line.

NB: If you're on a Windows machine, your mileage may vary...I still need to make better instructions for that case.

### b) Download the data
If you want to train a different property than formation energy per atom, replace it with its pymatgen string in that line.

Now, to download the structures, simply run:
```
python download_data.py "your_api_key"
```
where `"your_api_key"` is replaced with your actual key.

# 2. Train the network!
If you changed the property to train, make sure to update the file to reflect that (in the first example, this would be the line under "data-related options" reading `prop = "formation_energy_per_atom"`).

Otherwise, you should be able to simply run the the files in your Julia environment and see what happens!

(Note that you may get some CIF parsing warnings from pymatgen, but these shouldn't affect things and can be safely ignored)

Feel free to peruse and play with other options defined at the top of the file as well and see how it impacts the results! In particular, I've set the defaults to only have a dataset of size 100 so that the base case will run quickly, but feel free to try more data to see how much better the results get!
