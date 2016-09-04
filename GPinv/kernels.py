import tensorflow as tf
import GPflow
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

class SwitchedKernel(GPflow.kernels.Kern):
    """
    Block-wise kernel.
    The selection of the block is made by the extra dimension of X.
    """
    def __init__(self, kernel_list):
        """
        kernel_list should be a symmetric 2d-matrix.
        """
        GPflow.kernels.Kern.__init__(self, 1, 1)
        self.kernel_list = ParamList([ParamList(k_list) for k_list in kernel_list])
        self.num_kernel = len(kernel_list)

    def K(self, X, X2=None):
        # ind = X[:,-1]
        ind = tf.cast(tf.gather(tf.transpose(X), tf.shape(X)[1]-1), tf.int32)
        # X = X[:,:-1]
        X = tf.transpose(tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1]-1)))
        # split up X into chunks corresponding to the relevant likelihoods
        x_list = tf.dynamic_partition(X, ind, self.num_kernel)
        # partition for X
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_kernel)
        # prepare x2_list, ind2, partitions2
        if X2 is None:
            x2_list = x_list
            partitions2 = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_kernel)
        else:
            # ind2 = X2[:,-1]
            ind2 = tf.cast(tf.gather(tf.transpose(X2), tf.shape(X2)[1]-1), tf.int32)
            # split up X2
            X2 = tf.transpose(tf.gather(tf.transpose(X2), tf.range(0, tf.shape(X2)[1]-1)))
            x2_list = tf.dynamic_partition(X2, ind2, self.num_kernel)
            # partition for X2
            partitions2 = tf.dynamic_partition(tf.range(0, tf.size(ind2)), ind2, self.num_kernel)

        # calculate k and stitch togather
        k_mat = []
        for x, k2_list in zip(x_list, self.kernel_list):
            k_vec = []
            for x2, k in zip(x2_list, k2_list):
                # tf.shape(k_vec[i]) = [x2[i].shape[0], x.shape[0]]
                k_vec.append(k.K(x2, x))
            # tf.shape(k_mat[j]) = []
            k_mat.append(tf.transpose(tf.dynamic_stitch(partitions2, k_vec)))
        return tf.dynamic_stitch(partitions, k_mat)

    def Kdiag(self, X):
        # ind = X[:,-1]
        ind = tf.cast(tf.gather(tf.transpose(X), tf.shape(X)[1]-1), tf.int32)
        # X = X[:,:-1]
        X = tf.transpose(tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1]-1)))
        # split up X into chunks corresponding to the relevant likelihoods
        x_list = tf.dynamic_partition(X, ind, self.num_kernel)
        # apply the kernel to each section of the data
        results = [self.kernel_list[i][i].Kdiag(x_list[i]) for i in range(len(x_list))]
        # partition for X2
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_kernel)
        return tf.dynamic_stitch(partitions, results)


class BlockDiagonalKernel(SwitchedKernel):
    def __init__(self, kernel_list):
        """
        kernel_list is 1d list of kernel.
        """
        # generate block diagonal kernel_matrix with Zero kernel
        kernel_list2d = []
        for i in range(len(kernel_list)):
            kernel_list1d = []
            for j in range(len(kernel_list)):
                if i != j:
                    kernel_list1d.append(Zero(0))
                else:
                    kernel_list1d.append(kernel_list[i])
            kernel_list2d.append(kernel_list1d)
        SwitchedKernel.__init__(self, kernel_list2d)
