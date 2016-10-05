# GPinv
[![Build status](https://codeship.com/projects/8e8c5940-5322-0134-e799-4668b3c53a58/status?branch=master)](https://codeship.com/projects/147609)
[![Coverage status](https://codecov.io/gh/fujii-team/GPinv/branch/master/graph/badge.svg)](https://codecov.io/gh/fujii-team/GPinv)

A non-linear inverse problem solver with Gaussian Process prior.

## Background

We consider the following inverse problem,  
<img src=doc/readme_imgs/definition.png>  
where
**y** is the noisy observation of
some transform of the latent function **f**.  
**e** is noise component.
*F* is a function that is specific to the problem.  
For solving this inverse problem, a *prior* knowledge is necessary.  
In **GPinv**, we assume **f** follows *Gaussian Process*.

## Supported models
Currently, GPinv supports
+ Stochastic Variational Gaussian Process solver (StVGP).
+ Markov Chain Monte-Carlo method (GPMC).

The theoretical background for StVGP can be found in [Notebook (coming soon)](notebooks/StVGP_notes.ipynb)

For the usage, see the following examples,  
[Abel's inversion](notebooks/Abel_inversion.ipynb)  
[Spectroscopic extension of Abel's inversion](notebooks/Spectroscopic_Abel_inversion.ipynb)  


## Dependencies
**GPinv** heavily depends on
+ [**TensorFlow**](https://www.tensorflow.org/): a Large-Scale Machine Learning library.
+ [**GPflow**](https://github.com/GPflow/GPflow): a package for building Gaussian process models in python using TensorFlow.

Before installing **GPinv**, these two libreries must be installed.

For the TensorFlow installation,
see [**here**](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html).

For the GPflow installation, type
> `git clone https://github.com/GPflow/GPflow.git`  
> `cd GPflow`  
> `python setup.py develop`

For the GPinv installation, type
> `python setup.py install`
