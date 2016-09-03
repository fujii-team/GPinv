import numpy as np
import tensorflow as tf
import GPflow

class LinearModel(GPflow.model.GPModel):
    """
    Model for solving a linear inverse problem,
    where the data Y is a noisy observation of a linear transform of the latent
    function F.

    Y = A F + e
    """
    def __init__(self, X, Y, Amat, kern,
                                    mean_function=GPflow.mean_functions.Zero()):
        """
        :param 2d-np.array X: expressive data with shape [n, m]
        :param 2d-np.array Y: observation data with shape [N, m]
        :param 2d-np.array Amat: transformation matrix with shape [N, n]

        :param GPflow.kernels.Kern kern: GPflow's kernel object.
        :param GPflow.kernels.MeanFunction mean_function: GPflow's mean_function object.
        """
        GPflow.model.GPModel.__init__(self, X, Y, kern,
                    likelihood=GPflow.likelihoods.Gaussian(),
                    mean_function=mean_function)
        self.Amat = GPflow.param.DataHolder(Amat)


    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y, V | theta).

        Similar to GPflow.gpr.GPR.
        """
        I = GPflow.tf_hacks.eye(tf.shape(self.Y)[0])
        # K = A K A^T + e * I
        K = tf.matmul(
                tf.matmul(self.Amat, self.kern.K(self.X)),
              self.Amat, transpose_b = True) \
            + self.likelihood.variance * I

        L = tf.cholesky(K)
        # m = A m_F
        m = tf.matmul(self.Amat, self.mean_function(self.X))
        return GPflow.densities.multivariate_normal(self.Y, m, L)

    def build_predict(self):
        """

        """
        pass
