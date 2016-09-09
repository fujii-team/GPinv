# GPinv
[![Build status](https://codeship.com/projects/8e8c5940-5322-0134-e799-4668b3c53a58/status?branch=master)](https://codeship.com/projects/147609)
[![Coverage status](https://codecov.io/gh/fujii-team/GPinv/branch/master/graph/badge.svg)](https://codecov.io/gh/fujii-team/GPinv)

An inverse problem solver with Gaussian Process prior.

**Note**

This repository is currently for the **EDUCATIONAL** purpose for the software development.

## Background

We consider the following linear inverse problem,  
<img src=doc/readme_imgs/definition.png>  
where
**y** is the noisy observation of
some transform of the latent function **f**.  
**e** is independent zero-mean noise component.  
*F* is a function that is specific to the problem.  
For solving this inverse problem, a *prior* knowledge is necessary.  
In **GPinv**, we assume **f** follows *Gaussian Process*.

## Supported models
GPinv supports a linear model and some simple nonlinear models.

+ Linear model
[(Notebook)](notebooks/linear_model_example.ipynb)  
<img src=doc/readme_imgs/linear_model.png>  where A is a given matrix,
is supported.

+ Nonlinear model [(Notebook)](notebooks/nonlinear_model_example.ipynb)

  - MCMC with a single latent function.
  - VGP with a stochastic integration.  

In the future, GPinv will support more flexible models,
<img src=doc/readme_imgs/flexible_model.png> where observations are functions of multiple latent GP values.

More computationally efficient methods are also planned to be implemented.


## Dependencies
**GPinv** heavily depends on
[**TensorFlow**](https://www.tensorflow.org/) and
[**GPflow**](https://github.com/GPflow/GPflow).

Before installing **GPinv**, these two libreries must be installed.

For the TensorFlow installation,
see [**here**](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html).

For the GPflow installation, type
> `git clone https://github.com/GPflow/GPflow.git`  
> `cd GPflow`  
> `python setup.py develop`

For the GPinv installation, type
> `python setup.py install`
