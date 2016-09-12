from .mean_functions import Zero

class ModelInput(Object):
    """
    An object used for constructing a multi-latent models.
    """
    def __init__(self, X, kern, Z, mean_function=Zero(),
                        X_minibatch=False, minibatch_size=None, random_seed=0):
        """
        :param 2d-np.array X: Expressive coordinate
        :param Kern kern: GPinv.Kern object
        :param 2d-np.array Z: Inducing coordinate
        :param MeanFunction mean_functions: GPinv.MeanFunction object
        :param boolean X_minibatch: True if minibatching for X.
        :param integer minibatch_size: size of X-minibatch.
        :param integer random_seed: Random seed for X-minibatching.
        """
        self.X = X
        self.kern = kern
        self.Z = Z
        self.mean_function = mean_function
        self.X_minibatch = X_minibatch
        self.minibatch_size = minibatch_size
        self.random_seed = random_seed
