import tensorflow as tf
import numpy as np
from GPflow.likelihoods import Likelihood
from GPflow.priors import Gaussian
from GPflow.tf_wraps import eye
from GPflow.model import GPModel
from .param import Param, DataHolder
from .gpmc import TransformedGPMC
from .mean_functions import Zero
from .likelihoods import MultilatentLikelihood

class MultilatentGPMC(TransformedGPMC):
    """
    The same with GPflow.gpmc.GPMC, but can accept TransformedLikelihood.
    """
    """
    SVGP for the transformed likelihood with multiple latent functions.
    """
    def __init__(self, model_input_set,
                 Y, likelihood):
        """
        - model_inputs_set: ModelInputSet object
        - Y is a data matrix, size N' x R
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        """
        self.model_input_set = model_input_set
        self.num_latent = self.model_input_set.num_latent

        # Construct input vector, kernel, and mean_functions from input_list
        X = self.model_input_set.getConcat_X()
        Y = DataHolder(Y)

        kern          = self.model_input_set.getKernel()
        mean_function = self.model_input_set.getMeanFunction()

        # assert likelihood is appropriate
        assert isinstance(likelihood, MultilatentLikelihood)
        slice_begin, slice_end = self.model_input_set.generate_X_slices()
        likelihood.make_slice_indices(slice_begin, slice_end)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = self.model_input_set.num_latent
        self.V = Param(np.zeros((self.num_data, self.num_latent)))
        self.V.prior = Gaussian(0., 1.)

    # overwrite cholesky method for speeding up
    def getCholesky(self):
        return self.kern.Cholesky(self.X)
