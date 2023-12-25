# NAT Assignment 2023/2024

## About

Best assignment solutions for [Natural Computing 2023/2024](http://www.drps.ed.ac.uk/23-24/dpt/cxinfr11161.htm) offered by the [University of Edinburgh](https://www.ed.ac.uk/). The repository contains algorithm implementations for [Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) (and its variants, i.e., [Cuckoo Search](https://arxiv.org/abs/1003.1594v1), [Bat Algorithm](https://arxiv.org/abs/1308.3900), [Differential Evolution](https://link.springer.com/article/10.1023/A:1008202821328)), [Genetic Algorithm](https://arxiv.org/abs/2007.12673) (canonical), and [Genetic Programming](https://link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_24) (tree-based). Two main directories:

* `documents`: contains coursework description and my report
* `notebooks`: contains experiments for each of the questions

In addition, I attached my course [notes](documents/exam_notes.pdf). These are, however, squeezed into 6 pages which is what is allowed to be used during the _NOTES PERMITTED_ exam.

## Feedback

The coursework is worth 40% of the overall course mark, only the report is assessed but the code must be provided. Maximum points (**100**/100) were achieved with the following feedback:

> Fantastic study! Your coursework meets and exceeds expectations, demonstrating a quality that could be considered for publication. Your efforts and achievements are just excellent. Well done!

## Assignment Code

The source code (`src`) mainly contains classes and functions that model certain algorithms and problems. Notebooks focus on experiments with those algorithms and problems. In particular see:

* [problem1.ipynb](notebooks/problem1.ipynb): experiments for answering question 1 about particle cooperation
* [problem2.ipynb](notebooks/problem2.ipynb): experiments for answering question 2 about GA and Sumplete
* [problem3.ipynb](notebooks/problem3.ipynb): experiments for answering question 3 about GP and sequence generation

The experiment result files are not provided, however, they can all be reproduced by rerunning the cells (some may take several hours to run). There is actually no need to run the notebooks since the outputs are left displayed.

> Documentation is only written for the major classes

## Requirements

Please use at least _Python 3.10_. For rendering graphs for **Genetic Programming** question, please install [Graphviz](https://graphviz.org/). For example, if you're using _Ubuntu_:

```bash
sudo apt-get install graphviz
```

There are some package requirements as well. PLease install them as follows:

```bash
pip install -r requirements.txt
```

## Disclaimer

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

Other algorithm implementations were either covered in lectures or inspiration was taken from tutorials.