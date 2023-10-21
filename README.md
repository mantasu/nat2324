# NAT Assignment 2023/2024

## About

## Questions

Chosen functions (The difficulty can be varied by changing the dimension D):

Sphere Function: This is the simplest function as it’s unimodal (has only one minimum). The search space is smooth and doesn’t have local minima that could trap an optimization algorithm.

This is a simple unimodal function defined as:

$$f(\mathbf{x})=\sum_{d=1}^Dx_d^2$$

Rosenbrock Function: This function is more difficult than the Sphere function because it’s not separable, meaning the dimensions are interdependent. The function has a narrow, curved valley that contains the global minimum, and finding this valley can be challenging for an optimization algorithm.

$$f(\mathbf{x}) = \sum_{d=1}^{D-1} \left(100(x_{d+1}-x_d^2)^2 + (1 - x_d)^2\right)$$

Rastrigin function: The function is known for its large number of local minima, which makes it more difficult than the Sphere and Rosenbrock functions. The presence of the cosine term creates a complex, oscillating “landscape” with many local minima that can trap optimization algorithms.

$$f(\mathbf{x})=10 D+\sum_{d=1}^D\left[x_d^2-10 \cos \left(2 \pi x_d\right)\right]$$

Ackley Function: This function is more difficult than the Rosenbrock function because it has many local minima in addition to the global minimum. The presence of local minima can cause an optimization algorithm to get stuck and not find the global minimum.

$$f(\mathbf{x})=-20 \exp \left(-0.2 \sqrt{\frac{1}{D} \sum_{d=1}^D x_d^2}\right)-\exp \left(\frac{1}{D} \sum_{d=1}^D \cos \left(2 \pi x_d\right)\right)+20+e$$

Griewank Function: This function is more difficult than the Ackley function because it has many widespread local minima and a complex search space. The oscillations caused by the cosine term make it harder for an optimization algorithm to converge to the global minimum.

$$f(\mathbf{x})=\frac{1}{4000} \sum_{d=1}^D x_d^2-\prod_{d=1}^D \cos \left(\frac{x_d}{\sqrt{d}}\right)+1$$

Schwefel Function: This function is considered to be one of the most difficult among these five functions because it has a large number of local minima and a complex search space. The global minimum is at the bounds of the search space, which makes it particularly challenging for an optimization algorithm to find.

$$f(\mathbf{x})=418.9829 D-\sum_{d=1}^D x_d \sin \left(\sqrt{\left|x_d\right|}\right)$$

From easy to hard to optimize:

Sphere Function: This is typically considered the easiest to optimize. It’s a convex function and doesn’t have any local minima other than the global minimum. The gradient of this function points directly towards the minimum, so gradient-based optimization methods can solve it efficiently.

Rosenbrock Function: This function is more difficult because it has a narrow, curved valley that contains the global minimum. Optimization algorithms need to follow this valley to find the minimum, which can be challenging, especially for high-dimensional problems.

Ackley Function: The Ackley function has many local minima which can trap optimization algorithms. However, the difference between these local minima and the global minimum is relatively small, making it somewhat easier to optimize compared to functions with deep, isolated global minima.

Griewank Function: The Griewank function also has many local minima. The locations of these minima are sinusoidally modulated, which can make it difficult for optimization algorithms to navigate towards the global minimum.

Rastrigin Function: The Rastrigin function is known for its large number of local minima, which are regularly distributed throughout the search space. This makes it very easy for optimization algorithms to get stuck in a local minimum.

Schwefel Function: The Schwefel function is one of the most difficult benchmark functions to optimize. It has a large number of local minima that are irregularly distributed throughout the search space and far from the global minimum.

References:
* **Differential Evolution**: https://medium.com/@reshma_shaji/differential-evolution-what-it-is-and-how-does-it-work-81d3415c2367
* **Cuckoo Search**: https://github.com/ujjwalkhandelwal/cso_cuckoo_search_optimization/blob/main/cso/cso.py and https://medium.com/@evertongomede/cuckoo-search-algorithm-mimicking-nature-for-optimization-2fea1b96c82b
* **Bat Algorithm**: https://medium.com/it-paragon/bat-algorithm-bad-algorithm-b26ae42da8e1


## Problem 1

However, if we fix the number of generations rather than fitness evaluations, then we can expect the results to be better as more particles are searching for the global optimum, meaning there is a bigger likelihood one of them will hit it.

## Problem 2

If we have 3x3, we have in total 512 possibilities, meaning it's possible to just find a solution with brute force, which is essentially what the algorithm is doing as it acquires no knowledge of the problem as the time-steps pass.

## Problem 3

Problem 3 generality:

explain that using terminals of only integers/variables. We have background information and thus working with real numbers would only approach the solution, but may not solve it completely
imposed terminal/non-terminal constraints ()