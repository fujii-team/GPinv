import tensorflow as tf
import numpy as np
from GPflow import transforms
from GPflow.tf_wraps import eye
from GPflow import likelihoods
from . import densities
from .param import Param, Parameterized

from GPflow._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class Likelihood(Parameterized):
    """
    We newly implemented very simple likelihood, which requires only logp method.
    Additional method can be used for sampling.
    See sample_from method in StVmodel.
    """
    def __init__(self):
        Parameterized.__init__(self)
        self.scoped_keys.extend(['logp'])

    def logp(self,F,Y):
        """
        :param tf.tensor F: sized [N,n,R]
        :param tf.tensor Y: sized [k,m]
        where N is number of samples to approximate integration.
              n is number of evaluation point for one latent function,
              R is number of latent functions,
              k, m., Dimension of the data
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

    def F(self, F):
        return F

    def Y(self, F):
        return F + tf.sqrt(self.variance)*tf.random_normal(tf.shape(F), dtype=float_type)
