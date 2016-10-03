import tensorflow as tf
import numpy as np
import GPflow
from GPflow import kernels
from GPflow.tf_wraps import eye
from GPflow._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class Stationary(kernels.Stationary):
    """
    Multidimensional version of Stationary kernel.

    This kernel is sized [X.shape[0],X2.shape[0],R] and written by
    np.vstack([v_0*core(X,X2), v_1*core(X,X2), ..., v_R*core(X,X2)])

    This object provides efficient Cholesky Factorization method, self.Cholesky,
    where the cholesky tensor is
    np.vstack([sqrt(v_0)*chol, sqrt(v_1)*chol, ..., sqrt(v_R)*chol])
    with
    chol = Cholesky(K(X) + jitter)
    """
    def __init__(self, input_dim,
                 output_dim,
                 variance=None, lengthscales=None,
                 active_dims=None, ARD=False):
        """
        - input_dim is the dimension of the input to the kernel
        - output_dim is the dimension of the output of this kernel
                <-- This is an additional feature from GPflow.kernels.Stationary
        - variance : [1d-np.array] is the (initial) value for the variance parameter
                with size output_dim.
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        # variance should be 1d-np.array sized [output_dim]
        self.output_dim = output_dim
        if variance is None:
            variance = np.ones(output_dim)
        assert(variance.shape[0] == self.output_dim)
        kernels.Stationary.__init__(self, input_dim, variance, lengthscales,
                                    active_dims, ARD)
    def Kdiag(self,X):
        """
        Return: tf.tensor sized [N,R]
        """
        return tf.tile(tf.expand_dims(self.variance,0), [tf.shape(X)[0],1])

    def K(self, X, X2=None):
        core = tf.tile(tf.expand_dims(self._Kcore(X, X2),-1),
                                [1,1,tf.shape(self.variance)[0]]) # [N,N,R]
        var = tf.tile(
                tf.expand_dims(tf.expand_dims(self.variance, 0),0), # [1,1,R]
                    [tf.shape(core)[0],tf.shape(core)[1],1]) # [N,N,R]
        return var * core

    def Cholesky(self, X):
        core = self._Kcore(X, X2=None) + \
                    eye(tf.shape(X)[0]) * settings.numerics.jitter_level
        chol = tf.cholesky(core)
        var = tf.tile(tf.expand_dims(tf.expand_dims(
                            tf.sqrt(self.variance), 0),0),
                    [tf.shape(core)[0],tf.shape(core)[1],1])
        return var * tf.tile(tf.expand_dims(chol, -1),[1,1,tf.shape(var)[2]])

    def _Kcore(self, X, X2=None):
        """
        Returns
        """
        raise NotImplementedError

class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def _Kcore(self, X, X2=None):
        """
        """
        X, X2 = self._slice(X, X2)
        return tf.exp(-self.square_dist(X, X2)/2)
