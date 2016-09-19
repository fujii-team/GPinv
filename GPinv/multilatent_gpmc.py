import tensorflow as tf
import numpy as np
from GPflow.likelihoods import Likelihood
from GPflow.priors import Gaussian
from GPflow.tf_wraps import eye
from GPflow.model import GPModel
from .param import Param, DataHolder
from .gpmc import TransformedGPMC
from .mean_functions import Zero
from .multilatent_likelihoods import MultilatentLikelihood
from .multilatent_conditionals import conditional
from .multilatent_param import MultiFlow
from .multilatent_models import MultilatentModel

class MultilatentGPMC(MultilatentModel, TransformedGPMC):
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

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict
        This method computes
            p(F* | (F=LV) )
        where F* are points on the GP at Xnew, F=LV are points on the GP at X.
        """
        mu, var = conditional(Xnew, self.X, self.kern, self.V,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu + self.mean_function(Xnew), var

    @MultiFlow()
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        Return list of tensor.
        """
        fmu, var = self.build_predict(Xnew)
        return Xnew.partition(fmu), Xnew.partition(var)
