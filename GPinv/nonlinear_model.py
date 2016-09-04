import tensorflow as tf
import numpy as np
from GPflow.gpmc import GPMC
from likelihoods import NonLinearLikelihood
from kernels import BlockDiagonalKernel
from mean_functions import SwitchedMeanFunction

class NonlinearModel(object):
    method_list = ['gpmc', ]
    """
    Interface for the NonlinearModel.
    In this class, data and kernels are interpreted into the shape so that
    GPflow's model can handle them.

    The set of coordinate data (Xlist) is passed to the model as a long-vector
    with the list index is attached in the extra dimension.

    i.e.
    X[ 0:n0,:-1] = Xlist[0][ 0:n0], X[ 0:n0,-1] = 0
    X[n0:n1,:-1] = Xlist[0][n0:n1], X[n0:n1,-1] = 1
    ...

    The set of kernels and mean_functions are interpreted into the
    switched_kernel, and switched_mean_functions
    """
    def __init__(self, X, Y, kern, mean_function, likelihood,
                    method='gpmc', **options):
        """
        :param list of 2d-np.array Xlist: explanatory variable
                                (coordinates of parameters)
        :param list of 2d-np.array Y: observations.
        :param list of Kernel object: Kernel for each coordinate
        :param list of Mean object: mean_functions for each coordinate
        :param CorrelatedLikelihood object:
        :param string method: inference method, one of `method_list`
        """
        assert isinstance(likelihood, NonLinearLikelihood)
        assert method in self.method_list

        if isinstance(X, list):
            assert len(X) == len(kern) == len(mean_function)
            for x in X:
                assert X[0].shape[1] == x.shape[1]
                assert len(x.shape) == 2, "X should be 2-dimensional."
            # prepare the multiple vectors into one long-vector
            index_x=np.vstack([np.ones(len(X[i]), dtype=np.int32)*i
                                                    for i in range(len(X))])
            X = np.hstack([np.vstack(X), index_x.reshape(-1,1)])
            # set into kernels and means
            kernel = BlockDiagonalKernel(kernel)
            mean_function = SwitchedMeanFunction(mean_function)
        else:
            assert len(X.shape) == 2, "X should be 2-dimensional."
            index_x=np.zeros(X.shape[0])
            X = np.hstack([X, index_x.reshape(-1,1)])

        if isinstance(Y, list):
            # prepare the multiple vectors into one long-vector
            index_y=np.vstack([np.ones(len(Ylist[i]), dtype=np.int32)*i
                                                for i in range(len(Ylist))])
            num_latent = Y[0].shape[1]
        else:
            assert len(Y.shape) == 2, "Y should be 2-dimensional."
            index_y=np.zeros(Y.shape[0])
            num_latent = Y.shape[1]

        likelihood.setIndices(index_x, index_y)

        self.method = method
        if method == 'gpmc':
            self.model = GPMC(X, Y, kern, mean_function=mean_function,
                                          likelihood=likelihood,
                                          num_latent=num_latent)

    def optimize(self, *args):
        self.model.optimize(*args)

    def sample(self, num_samples, Lmax=20, epsilon=0.01, verbose=False,
                    return_logprobs=False, RNG=np.random.RandomState(0)):
        """
        Get samples from posterior based on HMC
        """
        return self.model.sample(num_samples, Lmax, epsilon, verbose,
                                    return_logprobs, RNG)
