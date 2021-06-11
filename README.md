## Description

This repo is to implement the method in this paper [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://arxiv.org/abs/1911.02590).

The authors' implementation can be found at this [link](https://github.com/lorraine2/implicit-hyper-opt).

The motivation to reimplement is to 
1. serve my own learning purpose
2. have a cleaner source code
3. compare several approaches related to inverse Hessian vector product


## Implementation detail

Suppose we have a model which is a subclass of ```nn.Module```, containing all parameters. ```BaseHypeOptModel``` in ```model.py``` will wrap this model and add hyperparameters. This ```BaseHyperOptModel``` will manage and intergate all the hyperparmaters in the main model such as computing the train loss via ```train_loss``` function, compute validation loss via ```validation_loss``` function. Currently ```BaseHyperOptModel``` allows its subclass to customize regularization and data augmentation. 

We introduce ```BaseHyperOptimizer``` object which compute hypergradient for hyperparameters via implicit function theorem. The subclass extending this object should provide a way to approximate inverse Hessian vector product. The current implementation contains serveral approaches for this
1. Conjugate Gradient
2. Neumann series expansion
3. Fixed point

This optimizer allows to choose between using Hessian matrix or Gauss-Newton Hessian matrix (see [this](https://proceedings.neurips.cc/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html)).

## Some useful references

1. [Hypertorch library](https://github.com/prolearner/hypertorch): An excellent library which this repo adopts in many parts. However, it's a little bit hard to work around with ```nn.Module.parameters```.
2. [Hyperparameter optimization with approximate gradient, ICML 2016](https://arxiv.org/abs/1602.02355): Maybe the first work of hyperparameter optimization using implicit gradient. Here the approximation tool is conjugate gradient method
3. [On the Iteration Complexity of Hypergradient Computation, ICML 2020](https://arxiv.org/pdf/2006.16218.pdf): In-depth comparison (convergence and approximate error) between iterative differentation (or unrolling) and approximate implicit differentation. The approximation considers two cases: fixed point vs conjugate gradient
4. [Convergence Properties of Stochastic Hypergradients, AISTATS 2021](https://arxiv.org/pdf/2011.07122.pdf): This work is quite important since previously we may blindly train implicit differentation method with minibatches of data and not know if it really converge.
5. [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://arxiv.org/abs/1911.02590): Approximate implicit differentation with Neumann series expansion. 
6. [Efficient and Modular Implicit Differentiation](https://arxiv.org/pdf/2105.15183.pdf): A recent work from Google explains a general, modular approach which modularizes solvers and autodiff.
7. [Roger Grosse's course](https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/): Excellent material for beginners from basic optimization to bilevel optimization.