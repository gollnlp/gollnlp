
# gollnlp - Grid Optimization Lawrence Livermore National Laboratory using NonLinear Programming

gollnlp is a code developed at Lawrence Livermore National Laboratory (LLNL) for the [ARPA-E Grid Optimization Competition Challenge 1](https://gocompetition.energy.gov/challenges/challenge-1). It scored favorably to the other partipating codes and ranked first in all four divisions of the competition. More information [here](https://gocompetition.energy.gov/kudos-livermore) and [here](https://gocompetition.energy.gov/challenges/challenge-1/leaderboards-final-event).

The gollnlp code solves security-constrained AC optimal flow (SC-ACOPF) problems in the so-called Challenge 1 Problem Formulation as specified [here](https://gocompetition.energy.gov/challenges/challenge-1/formulation). This page provides comprehensive information on the mathematical equations for SC-ACOPF, input data files, output format, evaluation scripts, scoring, rules, etc. 

The repository you are seeing here contains the competition submission only and it is not actively maintained. Please check the gollnlp project's [webpage](https://computing.llnl.gov/projects/gollnlp) for the latest capabilities and contact information. 

The computational approach implemented here is described in detail in [this](https://pubsonline.informs.org/doi/10.1287/opre.2022.0229) paper by Petra and Aravena. If you are refering to gollnlp, do not forget to cite the paper:
```
@article{petra23,
author = {Petra, Cosmin G. and Aravena, Ignacio},
title = {A Surrogate-Based Asynchronous Decomposition Technique for Realistic Security-Constrained Optimal Power Flow Problems},
journal = {Operations Research},
volume = {71},
number = {6},
pages = {2015-2030},
year = {2023},
doi = {10.1287/opre.2022.0229},
URL = {https://doi.org/10.1287/opre.2022.0229},
eprint = { https://doi.org/10.1287/opre.2022.0229 }
}
```

## Getting started

A standard build can be done by invoking the following in the 'build' directory 
```shell 
$> export IPOPT_PATH=/path/to/iptopbuild
$> cmake ..
$> make 
$> make test
$> make install
```



NOTE: requires that Ipopt was previously compiled as a static library
