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
