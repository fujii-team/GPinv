import tensorflow as tf
import numpy as np
import GPflow
from GPflow import kernels
from GPflow.tf_wraps import eye
from GPflow._settings import settings
from GPflow.param import ParamList
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class Kern(object):
    """
    An object that added to multi-dimensional functionality to
    GPflow.kernels.Kern.
    This object is meant to be inherited along with GPflow.kernels.Kern in child
    class.

    The main difference of this kernel from GPflow.kernels.Stationary is that
    this returns the multidimensional kernel values,
    sized [X.shape[0],X2.shape[0],R].

    The numpy equivalence is
    np.vstack([v_0*core(X,X2), v_1*core(X,X2), ..., v_R*core(X,X2)])

    This object provides efficient Cholesky Factorization method, self.Cholesky,
    where the cholesky tensor is
    np.vstack([sqrt(v_0)*chol, sqrt(v_1)*chol, ..., sqrt(v_R)*chol])
    with
    chol = Cholesky(K(X) + jitter)
    """
    def __init__(self, output_dim):
        """
        - input_dim is the dimension of the input to the kernel
        - output_dim is the dimension of the output of this kernel
                <-- This is an additional feature from GPflow.kernels.Stationary
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        """
        # variance should be 1d-np.array sized [output_dim]
        self.output_dim = output_dim

    def _Kcore(self, X, X2=None):
        """
        Returns the unit kernel which is common for all the output dimensions.
        """
        raise NotImplementedError


class Stationary(Kern, kernels.Stationary):
    """
    Multidimensional version of Stationary kernel.
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
        Kern.__init__(self, output_dim)
        # variance should be 1d-np.array sized [output_dim]
        if variance is None:
            variance = np.ones(output_dim)
        assert(variance.shape[0] == self.output_dim)
        kernels.Stationary.__init__(self, input_dim, variance, lengthscales,
                                    active_dims, ARD)

    def K(self, X, X2=None):
        core = tf.tile(tf.expand_dims(self._Kcore(X, X2),0),
                                [self.output_dim,1,1]) # [R,n,n]
        var = tf.expand_dims(tf.expand_dims(self.variance, -1),-1)
        return var * core # [R,n,n]

    def Kdiag(self,X):
        """
        Return: tf.tensor sized [N,R]
        """
        return tf.tile(tf.expand_dims(self.variance,-1), [1,tf.shape(X)[0]])

    def Cholesky(self, X):
        core = self._Kcore(X, X2=None) + \
                    eye(tf.shape(X)[0]) * settings.numerics.jitter_level
        chol = tf.cholesky(core)
        var = tf.expand_dims(tf.expand_dims(self.variance, -1),-1)
        return tf.sqrt(var) * tf.tile(tf.expand_dims(chol,0),[self.output_dim,1,1])

class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def _Kcore(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        return tf.exp(-self.square_dist(X, X2)/2)

class RBF_csym(RBF):
    """
    RBF kernel with a cylindrically symmetric assumption.
    The kernel value is
    K(x,x') = a exp(-(x+x)^2/2l^2)+a exp(-(x-x)^2/2l^2))
    """
    def _Kcore(self, X, X2=None):
        if X2 is None:
            X2 = X
        X = tf.abs(X)
        X2= tf.abs(X2)
        return RBF._Kcore(self, X, X2) + RBF._Kcore(self, X, -X2)

    def Kdiag(self, X):
        # returns [N] tensor
        X, _ = self._slice(X, None)
        X = tf.abs(X)
        square_dist = tf.reduce_sum(tf.square((X+X)/self.lengthscales), 1)
        # shape [R,N]
        diag = tf.exp(-0.5*square_dist)
        diag = tf.tile(tf.expand_dims(tf.ones_like(diag)+diag, 0),
                                                [self.output_dim, 1])
        var = tf.expand_dims(self.variance, -1)
        return var * diag

class RBF_casym(RBF):
    """
    RBF kernel with a cylindrically anti-symmetric assumption.
    The kernel value is
    K(x,x') = a exp(-(x-x)^2/2l^2)) - a exp(-(x+x)^2/2l^2)
    """
    def _Kcore(self, X, X2=None):
        if X2 is None:
            X2 = X
        X = tf.abs(X)
        X2= tf.abs(X2)
        return RBF._Kcore(self, X, X2) - RBF._Kcore(self, X, -X2)

    def Kdiag(self, X):
        # returns [N] tensor
        X, _ = self._slice(X, None)
        X = tf.abs(X)
        square_dist = tf.reduce_sum(tf.square((X+X)/self.lengthscales), 1)
        # shape [N,R]
        diag = tf.exp(-0.5*square_dist)
        diag = tf.tile(tf.expand_dims(tf.ones_like(diag)-diag, 0),
                                                    [self.output_dim, 1])
        var = tf.expand_dims(self.variance, -1)
        return var * diag

class Stack(Kern, kernels.Kern):
    """
    Kernel object that returns multiple kinds of kernel values, stacked
    vertically.

    Input for the initializer is a list of Kernel object, [k_1,k_2,...,k_M].
    The function call returns [k_1(X,X2),k_2(X,X2),...,k_M(X,X2)].
    The size of the return is n x n2 x (sum_i k_i.output_dim).
    """
    def __init__(self, list_of_kerns):
        """
        :param list list_of_kerns: A list of Kernel object.
        """
        output_dim = 0
        for k in list_of_kerns:
            # assert k is Kernel object
            assert(isinstance(k, Kern))
            output_dim += k.output_dim
        Kern.__init__(self, output_dim)
        kernels.Kern.__init__(self, input_dim=None)
        # kernels are stored as ParamList
        self.kern_list = ParamList(list_of_kerns)

    def K(self, X, X2=None):
        return tf.concat(0, [k.K(X,X2) for k in self.kern_list])

    def Kdiag(self,X):
        return tf.concat(0, [k.Kdiag(X) for k in self.kern_list])

    def Cholesky(self, X):
        return tf.concat(0, [k.Cholesky(X) for k in self.kern_list])
