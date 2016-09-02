import numpy as np
import tensorflow as tf
import GPflow.model.GPModel as GPModel
import GPflow.param.DataHolder as DataHolder

class LinearModel(GPflow.GPModel):
    """
    Model for solving a linear inverse problem,
    where the data Y is a noisy observation of a linear transform of the latent
    function F.

    Y = A F + e
    """
    def __init__(X, Y, Amat, kern, mean_function=GPflow.mean_functions.Zero()):
        """
        :param 2d-np.array X: expressive data with shape [n, m]
        :param 2d-np.array Y: observation data with shape [N, m]
        :param 2d-np.array Amat: transformation matrix with shape [N, n]

        :param GPflow.kernels.Kern kern: GPflow's kernel object.
        :param GPflow.kernels.MeanFunction mean_function: GPflow's mean_function object.
        """
        GPModel.__init__(self, X, Y, kern, mean_function)
        self.Amat = DataHolder(A)


    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y, V | theta).

        Similar to GPflow.gpr.GPR.
        """
        # K = A^T K A + e
        K = tf.matmul(
                tf.matmul(self.A, self.kern.K(self.X), transpose_a = True),
              self.A) \
            + self.likelihood.variance

        L = tf.cholesky(K)
        # m = A m_F
        m = self.matmul(A, self.mean_function(self.X))
        return multivariate_normal(self.Y, m, L)
