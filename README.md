# Monte Carlo Tree Search with Spectral Expansion for Planning with Dynamical Systems

Authors: Benjamin Riviere*, John Lathrop*, Soon-Jo Chung. * denotes Equal contribution. 

Data and source code are presented here for our Science Robotics paper: [Monte Carlo Tree Search with Spectral Expansion for Planning with Dynamical Systems](https://www.science.org/doi/10.1126/scirobotics.ado1010). This work develops and demonstrates the Spectral Expansion Tree Search (SETS) algorithm, which plans dynamical motions for robots using an efficient discrete representation. 

A video overview which accompanies our paper can be found below. It presents our hardware experiments which demonstrate that our algorithm can plan complex motions in real-time. The video also provides a visual summary of how our algorithm operates.

[Overview video](https://www.youtube.com/watch?v=o2ctFPs7OD4) 

## Citation

The data and code here are for personal and educational use only and provided without warranty; written permission from the authors is required for further use. Please cite our work as follows:

```
@article{
	doi:10.1126/scirobotics.ado1010,
	author = {Benjamin Rivière  and John Lathrop  and Soon-Jo Chung },
	title = {Monte Carlo tree search with spectral expansion for planning with dynamical systems},
	journal = {Science Robotics},
	volume = {9},
	number = {97},
	pages = {eado1010},
	year = {2024},
	doi = {10.1126/scirobotics.ado1010},
	URL = {https://www.science.org/doi/abs/10.1126/scirobotics.ado1010},
	eprint = {https://www.science.org/doi/pdf/10.1126/scirobotics.ado1010},
}
```


## Dryad

The primary data and code is stored in the `sets.tar.gz` tarball and can be extracted by `tar -zxvf sets.tar.gz` on linux systems. 

The directory contains the following subdirectories: `src` contains the solver and problem models in cpp and python bindings, `data` and `plots` stores the output of python scripts found in `scripts` with parameters specified in `configs`. We include two scripts: `value_convergence.py`, and `policy_convergence.py`, that should produce the plots in Figure 5 in the paper. There may be some small discreptancy in results because the solutions are anytime and therefore depend on machine-specific processor power. 

[https://doi.org/10.5061/dryad.s7h44j1h5](https://doi.org/10.5061/dryad.s7h44j1h5)

## Github

An up to date version of the code can be cloned from [https://github.com/aerorobotics/sets](https://github.com/aerorobotics/sets). Any updates will be pushed there. 


## Installation

Basic dependencies 
```bash
sudo apt install build-essential
sudo apt install libeigen3-dev
sudo apt install libyaml-dev
```

Use conda for most dependencies. 
```bash
conda env create --file environment.yml
```

Add src to path:
```
conda develop ~/src/
```

## Build
from `~/src/`, and with conda activated
```bash
mkdir build
cd build 
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release ..
make 
```

## Scripts

<!-- To make the rollout plot (fig 5a/b)
```
cd ~/scripts/
python rollout.py
```  -->

To make the value convergence plot (fig 5c)
```
cd ~/scripts/
python value_convergence.py
```

To make the policy convergence plot in the paper (fig 5d)
```
cd ~/scripts/
python policy_convergence.py
```
