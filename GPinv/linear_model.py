import numpy as np
import tensorflow as tf
from GPflow.tf_hacks import eye
import GPflow
from GPflow.param import AutoFlow

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

        self.num_latent = Y.shape[1]
        self.Amat = GPflow.param.DataHolder(Amat)


    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y, V | theta).

        Similar to GPflow.gpr.GPR.
        """
        # K = A K A^T + e * I
        K = tf.matmul(
                tf.matmul(self.Amat, self.kern.K(self.X)),
              self.Amat, transpose_b = True) \
            + self.likelihood.variance * eye(tf.shape(self.Y)[0])

        L = tf.cholesky(K)
        # m = A m_F
        m = tf.matmul(self.Amat, self.mean_function(self.X))
        return GPflow.densities.multivariate_normal(self.Y, m, L)


    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = tf.matmul(self.Amat, self.kern.K(self.X, Xnew))
        # K = A K A^T + e * I
        K = tf.matmul(
                tf.matmul(self.Amat, self.kern.K(self.X)),
              self.Amat, transpose_b = True) \
            + self.likelihood.variance * eye(tf.shape(self.Y)[0])
        L = tf.cholesky(K)
        # m = A m_F
        m = tf.matmul(self.Amat, self.mean_function(self.X))

        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - m)
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

    @AutoFlow()
    def predict_y(self):
        fmean, fvar = self.build_predict(self.X, full_cov=True)
        ymean = tf.matmul(self.Amat, fmean)
        # repeat number of latent_functions
        yvar = []
        for i in range(self.num_latent):
            yvar1 = tf.matmul(tf.matmul(self.Amat, fvar[:,:,i]),
                            self.Amat, transpose_b=True) \
                    +self.likelihood.variance * eye(tf.shape(self.Y)[0])
            yvar.append(tf.diag_part(yvar1))
        return ymean, tf.transpose(tf.pack(yvar))
