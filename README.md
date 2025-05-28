# Optimisers

This project implements a few simple optimisers suitable for quantum machine learning.

## Setup

It is advised to install this project directly through `pip`'s support of Git projects.

For example:
```bash
pip install git+https://github.com/we-taper/roto.git@A_Particular_Commit_Or_Brach_Or_Tag
```

E.g.

```bash
pip install git+https://github.com/we-taper/roto.git@v1.0.0
```

## Usage

The RotoSolver (paper [link](https://arxiv.org/abs/1903.12166)) is implemented and can be used as:

```python
from roto import RotoSolver

func = # the target function to be optimised
roto = RotoSolver()
var = # the initial parameter, usually randomly initiated.
print('Initial value:', func(var))
for _ in range(10): # try 10 iterations of RotoSolver
    var = roto.run_one_iteration(func=func, var=var)

print('Optimised value:', func(var))
```

Other optimisation algorithms implemented include: SGD, Adagrad, Adam, RProp, and various learning rate schedulers.
