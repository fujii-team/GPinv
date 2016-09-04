import tensorflow as tf
import numpy as np
from GPflow.gpmc import GPMC

class NonlinearModel(object):
    method_list = ['gpmc', 'vgp']
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
    def __init__(self, Xlist, Y, kernlist, meanlist, likelihood,
                    method='gpmc', **options):
        """
        :param list of 2d-np.array Xlist: explanatory variable
                                (coordinates of parameters)
        :param 2d-np.array Y: observations.
        :param list of Kernel object: Kernel for each coordinate
        :param list of Mean object: mean_functions for each coordinate
        :param CorrelatedLikelihood object:
        :param string method: inference method, one of `method_list`
        """
        assert len(Xlist) == len(kernlinst) == len(meanlist)
        self.kernlist, self.meanlist = kernlist, meanlist
        self.likelihood = likelihood
        # create the long-vector from Xlist
        X = np.hstack([
            # coordinate
            np.vstack(Xlist),
            # index is stored in the extra-dimension
            np.vstack([np.ones(len(Xlist[i]))*i for i in range(len(Xlist))])
        ])

        assert method in method_list
        self.method = method
        if method == 'gpmc':
            self.model = GPMC(X, Y,
                        )
