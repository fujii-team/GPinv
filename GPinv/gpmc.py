import tensorflow as tf
from GPflow.gpmc import GPMC
from GPflow.likelihoods import Likelihood
from GPflow.tf_wraps import eye
from .mean_functions import Zero
from .likelihoods import TransformedLikelihood

class TransformedGPMC(GPMC):
    """
    The same with GPflow.gpmc.GPMC, but can accept TransformedLikelihood.
    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=Zero(), num_latent=None):
        # assert likelihood is an instance of TransformedLikelihood
        assert isinstance(likelihood, TransformedLikelihood)
        GPMC.__init__(self, X, Y, kern, likelihood, mean_function, num_latent)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.
            \log p(Y, V | theta).
        """
        K = self.kern.K(self.X)
        L = tf.cholesky(K) + eye(tf.shape(self.X)[0])*1e-6
        F = tf.matmul(L, self.V) + self.mean_function(self.X)
        # TransformedLikelihood shoule have logp_gpmc method.
        return tf.reduce_sum(self.likelihood.logp_gpmc(F, self.Y))
