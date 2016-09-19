from functools import reduce
import tensorflow as tf
import GPflow
from GPflow.tf_wraps import eye
from .param import ParamList

class Zero(GPflow.kernels.Kern):
    """
    Zero kernel that simply returns the zero matrix with appropriate size
    """
    def __init__(self, input_dim, active_dims=None):
        GPflow.kernels.Kern.__init__(self, input_dim, active_dims)

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

class RBF_csym(RBF):
    """
    RBF kernel with a cylindrically symmetric assumption.

    The kernel value is

    K(x,x') = a exp(-(x+x)^2/2l^2)+a exp(-(x-x)^2/2l^2))
    """
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return RBF.K(self, X, X2) + RBF.K(self, X, -X2)

    def Kdiag(self, X):
        # returns [N] tensor
        X, _ = self._slice(X, None)
        square_dist = tf.reduce_sum(tf.square((X+X)/self.lengthscales), 1)
        return RBF.Kdiag(self, X) + \
                self.variance * tf.exp(-0.5*square_dist)


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

#---------------- Kernels for the multilatent model ------------------
class BlockDiagonal(GPflow.kernels.Kern):
    """
    The blockdiagonal kernel the element-matrix in which is given by kern_list
    ! NOTE !
    This kernel accepts param.ConcatParamList or param.ConcatDataHolder as
    arguments, rather than tf.tensor.
    """
    def __init__(self, kern_list, jitter=1.0e-4):
        GPflow.kernels.Kern.__init__(self, 1, 1)
        self.kern_list = ParamList(kern_list)
        self.jitter=jitter

    def K(self, X, X2=None):
        """
        :X and X2 ConcatParamList or ConcatDataHolder: expressive variable for K
        """
        if X2 is None:
            return reduce(tf.add,
                [tf.pad(k.K(x), [[begin, X.shape[0]-begin-tf.shape(x)[0]],
                                 [begin, X.shape[0]-begin-tf.shape(x)[0]]])
                    for k,x,begin
                    in zip(self.kern_list, X, X.slice_begin)])
        else:
            return reduce(tf.add,
                [tf.pad(k.K(x,x2 ), [[begin, X.shape[0] -begin -tf.shape(x)[0]],
                                     [begin2,X2.shape[0]-begin2-tf.shape(x2)[0]]])
                    for k,x,begin,x2,begin2
                    in zip(self.kern_list, X, X.slice_begin,X2,X2.slice_begin)])

    def Kdiag(self, X):
        return reduce(tf.add,
            [tf.pad(k.Kdiag(x), [[begin,X.shape[0]-begin-tf.shape(x)[0]]])
                for k,x,begin
                in zip(self.kern_list, X, X.slice_begin)])

    def Cholesky(self, X):
        """
        Compute the cholesky decomposition for K(X).
        Since this kernel is block diagonal, it can be computed very efficiently.
        """
        return reduce(tf.add,
            [tf.pad(
                tf.cholesky(k.K(x) + self.jitter*eye(tf.shape(x)[0])),
                    [[begin,X.shape[0]-begin-tf.shape(x)[0]],
                     [begin,X.shape[0]-begin-tf.shape(x)[0]]])
                for k,x,begin
                in zip(self.kern_list, X, X.slice_begin)]
            )
