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
    Block-wise kernel.
    The selection of the block is made by the extra dimension of X.
    """
    def __init__(self, kern_list,
                        slice_X_begin, slice_X_size,
                        slice_X2_begin, slice_X2_size, jitter=0.):
        """
        - kern_list : list of Kernels.
        """
        GPflow.kernels.Kern.__init__(self, 1, 1)
        self.kern_list = ParamList(kern_list)
        # set the slice indices
        self.slice_X_begin, self.slice_X_size = slice_X_begin, slice_X_size
        self.slice_X2_begin, self.slice_X2_size = slice_X2_begin, slice_X2_size
        self.jitter = jitter

    def K(self, X, X2=None):
        if X2 is None:
            return reduce(tf.add,
                [tf.pad(k.K(tf.slice(X, [begin,0], [size, -1])),
                        [[begin,tf.shape(X)[0]-begin-size],
                         [begin,tf.shape(X)[0]-begin-size]])
                    for k,begin,size
                    in zip(self.kern_list, self.slice_X_begin, self.slice_X_size)])
        else:
            return reduce(tf.add,
                [tf.pad(k.K(tf.slice(X, [begin,0], [size, -1]),
                            tf.slice(X2,[begin2,0],[size2,-1])),
                        [[begin,tf.shape(X)[0]-begin-size],
                         [begin2,tf.shape(X2)[0]-begin2-size2]])
                    for k,begin,size,begin2,size2
                    in zip(self.kern_list, self.slice_X_begin, self.slice_X_size,
                                           self.slice_X2_begin,self.slice_X2_size)])

    def Kdiag(self, X):
        return reduce(tf.add,
            [tf.pad(k.Kdiag(tf.slice(X, [begin,0], [size, -1])),
                    [[begin,tf.shape(X)[0]-begin-size]])
                for k,begin,size
                in zip(self.kern_list, self.slice_X_begin, self.slice_X_size)])

    def Cholesky(self, X):
        """
        Compute the cholesky decomposition for K(X).
        Since this kernel is block diagonal, it can be computed very efficiently.
        """
        return reduce(tf.add,
            [tf.pad(
                tf.cholesky(k.K(tf.slice(X, [begin,0], [size, -1])) + self.jitter*eye(size)),
                    [[begin,tf.shape(X)[0]-begin-size],
                     [begin,tf.shape(X)[0]-begin-size]])
                for k,begin,size
                in zip(self.kern_list, self.slice_X_begin, self.slice_X_size)]
            )
