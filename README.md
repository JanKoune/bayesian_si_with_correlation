# Bayesian system identification for structures considering spatial and temporal correlation

This repository contains the scripts to run the synthetic case described in the paper "Bayesian system identification for structures considering spatial and temporal correlation"

## Folder structure

 - /analysis/: This is the main folder containing the definitions of the analyses
 - /continuous_girder/: Julia FE model of the IJssel bridge
 - /data/: Saved data from bayesian inference runs
 - /figures/: Manually saved and automatically generated figures
 - /measurements/: Synthetic measurement data
 - /models/: FE models written in Python utilizing the Julia FE models.
 - /modules/: Various utilities

## Usage

1. Install the required packages using `pip install -r requirements.txt`
2. Install the [taralli](https://gitlab.com/tno-bim/taralli) Python package
3. Install the [PyJulia](https://pyjulia.readthedocs.io/en/latest/index.html) Python package
4. Set the `path_julia` variable in `modules/__init__.py` to point to a Julia installation as described in the inline comment
5. The code for performing the analyses should be executed with the working directory set to the root of this directory

## Notes

* [Julia](https://julialang.org/) is required to use the IJsselbridge FE model 
* The scripts in this repository have been tested with Julia 1.8.3 and Python 3.10