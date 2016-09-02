# GPinv
[![travisCI](https://travis-ci.org/fujii-team/GPinverse.svg?branch=master)](https://travis-ci.org/)
[![codecov](https://codecov.io/gh/fujii-team/GPinverse/branch/master/graph/badge.svg)](https://codecov.io/gh/fujii-team/GPinverse)

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
We assume **f** follows *Gaussian Process*.

## Supported models
Currently, only linear model with
<img src=doc/readme_imgs/linear_model.png>  where A is a given matrix.
is supported.

In the future, GPinverse may support
non-linear models.


## Dependencies
**GPinv** is heavily depends on
[**GPflow**](https://github.com/GPflow/GPflow)
and [**TensorFlow**](https://www.tensorflow.org/).

For the TensorFlow installation,
see [**here**](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html).

For the GPflow installation, type
> `git clone https://github.com/GPflow/GPflow.git`  
> `cd GPflow`  
> `python setup.py install`