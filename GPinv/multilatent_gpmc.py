import tensorflow as tf
from GPflow.likelihoods import Likelihood
from GPflow.tf_wraps import eye
from .param import DataHolder
from .gpmc import TransformedGPMC
from .mean_functions import Zero
from .likelihoods import TransformedLikelihood


class MultilatentGPMC(TransformedGPMC):
    """
    The same with GPflow.gpmc.GPMC, but can accept TransformedLikelihood.
    """
    """
    SVGP for the transformed likelihood with multiple latent functions.
    """
    def __init__(self, input_list,
                 Y, likelihood, num_latent=None):
        """
        - model_inputs: list of ModelInput objects.
        - Y is a data matrix, size N' x R
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_shape is one of ['fullrank', 'diagonal', 'specified']
        - q_indices_list is list of tuples, which indicates the corelation
                                                between each model_input.
        - minibatch_size is the size for the minibatching for Y
        - random_seed is the seed for the Y-minibatching.
        """
        self.input_list = input_list
        self.num_latent = num_latent or Y.shape[1]

        # Construct input vector, kernel, and mean_functions from input_list
        X = IndexedDataHolder(self.input_list)
        Y = DataHolder(Y)

        kern          = BlockDiagonalKernel([d.kern          for d in input_list])
        mean_function = SwitchedMeanFunction([d.mean_function for d in input_list])
        # assert likelihood is appropriate
        assert isinstance(likelihood, MultilatentLikelihood)
        likelihood.make_slice_indices(self.input_list)

        # init the super class, accept args
        TransformedSVGP.__init__(self, X, Y, kern, likelihood, mean_function)
