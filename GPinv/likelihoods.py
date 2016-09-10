import tensorflow as tf
import numpy as np
from GPflow.likelihoods import Likelihood
from GPflow import transforms
from GPflow.param import Param, DataHolder
from GPflow.likelihoods import Likelihood
from GPflow import densities
from GPflow import svgp

class MinibatchData(svgp.MinibatchData):
    pass

class TransformedLikelihood(Likelihood):
    def __init__(self, num_samples=20):
        """
        Likelihood with correlation.
        :param num_samples: number of random point to approximate the
                                     variational expectation.
        """
        Likelihood.__init__(self)
        # number of random numbers to approximate the integration
        self.num_samples = num_samples

    def stochastic_expectations(self, Fmu, L, Y):
        """
        Evaluate variational expectation based on the stochastic method.
        We assume the likelihood is independent to some transform of the latent
        functions
         \integ{logp(Y|f) N(f|Fmu,LLt) df}
         = logp(Y|g(f))
         = \sum_{i} \integ{log(Y_i, g_i)p(g_i)g_i}

        We approximate p(g_i) by a Gaussian function, whose mean and variance
        is estimated from the stochastic method.

        :args
         Fmu: Mean of the latent function. shape=[N,M]
         L  : Cholesky of covariance. shape=[N,N,M]
         Y  : Data. shape=[N',M']

        :return
         Stochastic approximation of
         \integ{logp(Y|f) N(f|Fmu,LLt) df}
        """
        # normal random vector with shape [num_samples, M, N, 1]
        rndn = tf.random_normal(
                    [self.num_samples, tf.shape(L)[2], tf.shape(L)[1], 1],
                    dtype=tf.float64)
        # L.shape [num_samples, M, N, N]
        L = tf.tile(tf.expand_dims(tf.transpose(L, [2,0,1]), [0]),
                                    [self.num_samples, 1,1,1])
        # Sampled point of F.
        # X.shape = [num_samples, N, M]. Mean: Fmu, Cov: LLt
        X = tf.tile(tf.expand_dims(Fmu, [0]), [self.num_samples, 1, 1]) + \
            tf.transpose(tf.squeeze(tf.batch_matmul(L, rndn), [3]), [0,2,1])
        # The mean and variance of latent values for data point Y.
        # Shape [num_samples', N', M']
        G = self.transform(X)
        # Estimate mean and variance. shape [N, M] for each.
        Gmu = tf.reduce_mean(G, reduction_indices=[0], keep_dims=True)        # shape [1, N', M']
        # shape [N', M']
        Gvar = tf.reduce_mean(
            tf.square(G-tf.tile(Gmu, [self.num_samples, 1, 1])),
            reduction_indices=[0])
        Gmu = tf.squeeze(Gmu,[0])

        return self.variational_expectations(tf.transpose(Gmu), tf.transpose(Gvar), Y)

    def logp_gpmc(self, X, Y):
        """
        A method that for the gpmc.
        :param tf.tensor X: shape [N, M]
        :param tf.tensor Y: shape [N',M']
        """
        # reshape to match self.transform input
        X = tf.expand_dims(X, [0])
        G = self.transform(X)
        # reshape to match self.logp input
        G = tf.squeeze(G, [0])
        return self.logp(G, Y)

    def transform(self, F):
        """
        Transform of the latent functions to latent values.
        :param tf.tensor F: Samples for latent functions sized [num_samples, M, N]
        :return tf.tensor: Samples for the latent values at Y, sized
                                                        [num_samples', M', N']
        The return value should be real and vary around some values.
        E.G. the latent values that is limited to positive space should be
        projected into the entire real space by Log function.
        """
        raise NotImplementedError

    def logp(self, F, Y):
        """
        Return p(Y|F)

        :param tf.tensor Y: shape [N', M']
        :param tf.tensor F: shape [N', M']
        """
        raise NotImplementedError


class Gaussian(TransformedLikelihood):
    """
    i.i.d Gaussian with uniform variance.
    Stochastic expectation is used.
    """
    def __init__(self, num_samples=20, exact=True):
        """
        :param bool exact: If True then analytically calculate
                                            stochastic_expectations.
        """
        TransformedLikelihood.__init__(self, num_samples)
        self.variance = Param(1.0, transforms.positive)
        self.exact = exact

    def transform(self, F):
        return F

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)

    def stochastic_expectations(self, Fmu, L, Y):
        if self.exact: # exact calculation
            L = tf.transpose(L, [2,0,1])
            Fvar = tf.batch_matrix_diag_part(tf.batch_matmul(L, L, adj_y=True))
            Fmu = tf.transpose(Fmu)
            Y = tf.transpose(Y)
            return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
                   - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance
        else:
            return TransformedLikelihood.stochastic_expectations(self, Fmu, L, Y)

class MinibatchGaussian(Gaussian):
    def __init__(self, num_data, minibatch_size, num_samples=20, exact=True):
        Gaussian.__init__(self, num_samples, exact)
        # transfer function
        self.I = MinibatchData(np.eye(num_data), minibatch_size)

    def transform(self, F):
        I = tf.tile(tf.expand_dims(self.I, [0]), [tf.shape(F)[0], 1, 1])
        return tf.batch_matmul(I, F)

'''
class NonLinearLikelihood(StochasticLikelihood):
    """
    Likelihood for the nonlinear_model.
    This is a wrap for StochasticLikelihood.
    """
    def setIndices(self, index_x, index_y):
        """
        Setting the index for dividing the latent function samples.

        :param 1d-np.array(int) index_x: list of index value for the latent
            functions.
        :param 1d-np.array(int) index_y: list of index value for the observation.
        """
        self.index_x = DataHolder(index_x.astype(np.int32))
        self.index_y = DataHolder(index_y.astype(np.int32))
        self.num_x = np.int(np.max(index_x)) + 1
        self.num_y = np.int(np.max(index_y)) + 1

    def logp(self, X, Y):
        """
        Wrap logp(self, X, Y) for the convenience.
        """
        if self.num_x==1:
            Xlist = [X,]
        else:
            Xlist = tf.dynamic_partition(X, self.index_x, self.num_x)

        if self.num_y==1:
            Ylist = [Y,]
            return self.log_prob(Xlist, Ylist)[0]
        else:
            Ylist = tf.dynamic_partition(Y, self.index_y, self.num_y)
            # partitionk
            partitions = tf.dynamic_partition(tf.range(0, tf.size(self.index_y)),
                                                        self.index_y, self.num_y)
            return tf.dynamic_stitch(partitions, self.log_prob(Xlist, Ylist))

    def batch_logp(self, X, Y):
        """
        Wrap batch_logp(self,X,Y) for the convenience.
        """
        Xlist = tf.dynamic_partition(X, self.index_x, self.num_x)
        Ylist = tf.dynamic_partition(Y, self.index_y, self.num_y)
        # partitionk
        partitions = tf.dynamic_partition(tf.range(0, tf.size(self.index_y)),
                                                    self.index_y, self.num_y)
        if self.num_y==1:
            return self.batch_log_prob(Xlist, Ylist)[0]
        else:
            return tf.dynamic_stitch(partitions, self.batch_log_prob(Xlist, Ylist))

    def log_prob(self, Xlist, Ylist):
        """
        This part should be implemented in the child class.
        :param list of tensor Xlist: list of the latent functions with length Q.
                The shape of the i-th element is [Ni,M]
        :param list of tensor Ylist: list of the observations with length P.
                The shape of the i-th element is [Ni',M']
        :return list of log of the likelihood with length P.
            The shape should be the same to that of Ylist.
        """
        raise NotImplementedError

    def batch_log_prob(self, Xlist, Ylist):
        """
        This part should be implemented in the child class.
        :param list of tensor Xlist: list of the latent functions with length Q.
                The shape of the i-th element is [Ni,M, num_stocastic_points]
        :param list of tensor Ylist: list of the observations with length P.
                The shape of the i-th element is [Ni',M', num_stocastic_points]
        :return list of log of the likelihood with length P.
            The shape should be the same to that of Ylist.
        """
        raise NotImplementedError
'''
