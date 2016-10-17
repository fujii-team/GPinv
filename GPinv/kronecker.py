import tensorflow as tf
import numpy as np
from GPflow.tf_wraps import eye
from . param import DataHolder
from . import kernels
from GPflow._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

def kronecker_product(A, B):
    """
    Returns the Kronecker's product between A and B
    """
    m = tf.shape(A)[0]
    n = tf.shape(A)[1]
    p = tf.shape(B)[0]
    q = tf.shape(B)[1]
    A_tiled = tf.reshape(
                tf.tile(tf.expand_dims(tf.expand_dims(A,1), 3), [1,p,1,q]),
                [m*p,n*q])
    B_tiled = tf.tile(B, [m,n])
    return A_tiled*B_tiled # shape [m*p, n*q]

class RBF2D(kernels.RBF):
    def __init__(self, input_dim,
         output_dim,
         dim1, dim2,
         variance=None, lengthscales=None,
         active_dims=None, ARD=False):
        """
        - input_dim is the dimension of the input to the kernel
        - output_dim is the dimension of the output of this kernel
        - dim1, dim2: X coordinate for dim1 and dim2
        - variance : [1d-np.array] is the (initial) value for the variance parameter
                with size output_dim.
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        kernels.RBF.__init__(self, input_dim, output_dim, variance, lengthscales,
                    active_dims, ARD)
        # axis data is stored
        self.dim1 = DataHolder(dim1, 'recompile')
        self.dim2 = DataHolder(dim2, 'recompile')

    def Cholesky(self, X):
        """
        Overwrite cholesky for the speed up.
        X should be dim2*dim2
        """
        chol_dim1 = tf.cholesky(
                self._Kcore(self.dim1, X2=None) + \
                eye(tf.shape(self.dim1)[0]) * settings.numerics.jitter_level)
        chol_dim2 = tf.cholesky(
                self._Kcore(self.dim2, X2=None) + \
                eye(tf.shape(self.dim2)[0]) * settings.numerics.jitter_level)
        # core of the cholesky
        chol = kronecker_product(chol_dim1, chol_dim2)
        # expand and tile
        var = tf.tile(tf.expand_dims(tf.expand_dims(
                            tf.sqrt(self.variance), 0),0),
                    [tf.shape(chol)[0],tf.shape(chol)[1],1])
        return var * tf.tile(tf.expand_dims(chol, -1),[1,1,tf.shape(var)[2]])
