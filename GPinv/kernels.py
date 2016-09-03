import GPflow

class Zero(GPflow.kernels.Kern):
    """
    Zero kernel that simply returns the zero matrix with appropriate size
    """
    def __init__(self, input_dim, active_dims=None):
        Kern.__init__(self, input_dim, active_dims)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return tf.zeros(tf.pack([tf.shape(X)[0], tf.shape(X2)[0]]), dtype=tf.float64)

    def Kdiag(self, X):
        return tf.zeros(tf.pack([tf.shape(X)[0]]), dtype=tf.float64)


class White(GPflow.kernels.White):
    """  Identical to GPflow.kernels.Constant    """
    pass

class Constant(GPflow.kernels.Constant):
    """  Identical to GPflow.kernels.Constant    """
    pass

class Bias(GPflow.kernels.Bias):
    """  Identical to GPflow.kernels.Bias    """
    pass

class RBF(GPflow.kernels.RBF):
    """  Identical to GPflow.kernels.RBF    """
    pass

class RBF_csym(GPflow.kernels.Stationary):
    """
    RBF kernel with a cylindrically symmetric assumption.
    """
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return RBF.K(self, X, X) + RBF.K(self, X, -X)

    def Kdiag(self, X):
        # TODO
        pass


class Linear(GPflow.kernels.Linear):
    """  Identical to GPflow.kernels.Linear    """
    pass

class Exponential(GPflow.kernels.Exponential):
    """  Identical to GPflow.kernels.Exponential    """
    pass

class Matern12(GPflow.kernels.Matern12):
    """  Identical to GPflow.kernels.Matern12    """
    pass

class Matern32(GPflow.kernels.Matern32):
    """  Identical to GPflow.kernels.Matern32    """
    pass

class Matern52(GPflow.kernels.Matern52):
    """  Identical to GPflow.kernels.Matern52    """
    pass

class Cosine(GPflow.kernels.Cosine):
    """  Identical to GPflow.kernels.Cosine    """
    pass

class Coregion(GPflow.kernels.Coregion):
    """  Identical to GPflow.kernels.Coregion    """
    pass
