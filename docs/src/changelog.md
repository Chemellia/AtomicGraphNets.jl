# Changelog

I'm generally trying to adhere to [semver](https://semver.org) here. This means that in v0.*, assume breaking changes are always possible, even without a major version bump...however, I will try to always note them here if they happen...

Categories to include for each release, if relevant: breaking, added, fixed, removed/deprecated

## Upcoming


## v0.2.0 [2021-07-14]
Identical to v0.1.3, tagged for some Pkg convenience purposes.

## v0.1.4 [2021-07-14]
Identical to v0.1.2 but with Flux compat entry included both v0.11 and v0.12.

## v0.1.3 [2021-07-09]
### Breaking
* now compatible with latest ChemistryFeaturization API

## v0.1.2 [2021-03-22]

### Added:
* CGCNN model builder defaults to half `atom_conv_fea_len` for `pooled_fea_len` parameter
* docs: add documentation website!
* docs: remove conda environment files from example 1 and just include instructions to set up an environment from scratch for downloading data

### Fixed: 
* import loss functions via `Flux.Losses` to be compatible with upcoming Flux v0.12
* change syntax of `build_graphs_batch` in example 2 to match changes to ChemistryFeaturization
* rename `master` branch to `main`

## v0.1.1 [2021-03-11]
### Fixed:
* explicitly import `Flux.glorot_uniform`, which is no longer exported by Flux otherwise

## v0.1.0 [2021-02-25]

First released version!

### Added:
* create `AGNConv` and `AGNPool` layers
* create basic CGCNN model builder
