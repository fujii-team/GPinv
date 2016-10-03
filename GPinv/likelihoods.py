import tensorflow as tf
import numpy as np
from GPflow import transforms
from GPflow.tf_wraps import eye
from GPflow import likelihoods
from . import densities
from .param import MinibatchData, Param, DataHolder, Parameterized

class Likelihood(Parameterized):
    """
    We newly implemented very simple likelihood, which requires only logp method.
    """
    def __init__(self):
        Parameterized.__init__(self)
        self.scoped_keys.extend(['logp'])

    def logp(self,F,Y):
        """
        Return the log density of the data given the function values.
        """
        raise NotImplementedError("implement the logp function\
                                  for this likelihood")

class Gaussian(Likelihood):
    def __init__(self):
        Likelihood.__init__(self)
        self.variance = Param(1.0, transforms.positive)

    def logp(self, F, Y):
        """
        :param F tf.tensor(N,n,R): Sample from the posterior
        :param Y tf.tensor(n,R): Observation
        """
        Y = tf.tile(tf.expand_dims(Y, 0), [tf.shape(F)[0],1,1])
        return densities.gaussian(F, Y, self.variance)
