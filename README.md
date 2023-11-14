# NAT Assignment 2023/2024

## About


## Problem 1

However, if we fix the number of generations rather than fitness evaluations, then we can expect the results to be better as more particles are searching for the global optimum, meaning there is a bigger likelihood one of them will hit it.

## Problem 2

If we have 3x3, we have in total 512 possibilities, meaning it's possible to just find a solution with brute force, which is essentially what the algorithm is doing as it acquires no knowledge of the problem as the time-steps pass.

## Problem 3

Problem 3 generality:

explain that using terminals of only integers/variables. We have background information and thus working with real numbers would only approach the solution, but may not solve it completely
imposed terminal/non-terminal constraints ()

## Setup

```bash
sudo apt-get install graphviz
```

## Disclaimer

### Reverse ranges

In the experiments where `num_evaluations` instead of `max_generations` is used, the range of the population size is reversed, i.e., instead of $\{1, ..., N\}$, we have $\{N, ..., 1\}$ (of course, when plotting, this makes no difference because values are ordered by default).

This is because the experiments are parallelized and the progress bar is updated in the sequence the processes have started, meaning if the first process takes very long to run the experiment and the other ones take very little time to ron other experiments, the progress bar will still only update after the first process finishes, and it will update to `100%` immediately since there would be no other processes to wait. So thus we reverse to see the updates of the progress bar more frequently because it is the case that the first experiments run much longer than the last ones (see below why). This can be considered as an issue with `tqdm` package, specifically, `process_map` method.

Regardless of how many individuals there are (unless $N$ is very large), it is much faster evolve less generations with more individuals than to evolve more generations with less individuals. This is because at each iteration of `evolve`, all the functions are vectorized using `numpy` which uses `C++` under the hood. Now if we run many generations with few individuals, this will be slow because running many _Python_ loops is very slow.

### Rerunning previous cells

Please also note that the cells are designed to be run in sequence and no previous cell is expected to be rerun. This is because most of teh subsequent cells overwrite the variables created by the previous cells (for easier readability, variable names are intended to be kept the same throughout the notebooks for every experiment). So, for example, instead of initializing `N_experiment_1`, `N_experiment_2`, etc., the same variable `N` is refreshed (with possibly different values) in every subsequent cell.

Although this may not be a good practice for long experiments (due to the results possibly getting lost), this is still intended because all the experiments are saved and can be quickly reloaded, in case we need to rerun the cells (e.g., if we change the style of the plots).

### Saving experiment runs

Experiments will be automatically saved as `.npz` files with names generated from variable and static arguments, i.e., based on parameters that are being experimented with and the default ones. Note that, although for static arguments values are also labeled, e.g., if a static parameter is `static_param["num_evaluations"] = 5000`, then the filename will match that, i.e., `num_evaluations=5000.npz`, the same is not done for variable parameters (otherwise there would be too many values), only the parameter names are included, e.g., if a variable parameter is `variable_param['N'] = range(1, 101)`, then the filename will match only the variable name, i.e., `[N].npz`.

For this reason if multiple experiments are performed with different variable parameters, please note that the saved files may override the other ones. So either include some unique ranges for every experiment that uses the same variable parameters or include a dummy label in a static parameters dictionary, e.g., `static_param["version"]=1`.

## Issues

If the notebooks cannot be run due to failed imports, i.e., because `sys.path.append` does not work in an expected way, please perform the following steps:

1. Move `src` directory tree inside `notebooks` folder. The file layout should now look as follows:
   ```bash
   └── path/to/nat2324
     ├── notebooks
     |    ├── src/nat2324       # The source code (folders and python files)
     |    ├── problem1.ipynb    # Problem 1 notebook
     |    ├── problem2.ipynb    # Problem 2 notebook
     |    └── problem3.ipynb    # Problem 3 notebook
     └── ...                    # Other files and folders except `src`
   ``` 
2. Prepend `src.` before imports from `nat2324`, e.g., the setup cell in `problem1.ipynb` would now look as follows:
    ```python
    import numpy as np
    from src.nat2324.problems import Objective
    from src.nat2324.algorithms import SwarmOptimization
    from src.nat2324.utils import *
    ```

## References

Some of the algorithm implementations (particularly in **problem 1**) took inspiration from existing repositories, or online tutorials:

* **Differential Evolution**: https://medium.com/@reshma_shaji/differential-evolution-what-it-is-and-how-does-it-work-81d3415c2367
* **Cuckoo Search**: https://github.com/ujjwalkhandelwal/cso_cuckoo_search_optimization/blob/main/cso/cso.py and https://medium.com/@evertongomede/cuckoo-search-algorithm-mimicking-nature-for-optimization-2fea1b96c82b
* **Bat Algorithm**: https://medium.com/it-paragon/bat-algorithm-bad-algorithm-b26ae42da8e1