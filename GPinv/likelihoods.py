import tensorflow as tf
import numpy as np
from GPflow.likelihoods import Likelihood
from GPflow import transforms
from GPflow.param import Param, DataHolder
from GPflow.likelihoods import Likelihood
from GPflow import densities

class StochasticLikelihood(Likelihood):
    def __init__(self, num_stocastic_points=20):
        """
        Likelihood with correlation.
        :param 2-element-list shape: shape of variables passed to this likeilhood.
        :param num_stocastic_points: number of random point to approximate the
                                     variational expectation.
        """
        Likelihood.__init__(self)
        # number of random numbers to approximate the integration
        self.num_stocastic_points = num_stocastic_points

    def stochastic_expectations(self, Fmu, L, Y):
        """
        Evaluate variational expectation based on the stochastic method.
        :args
         Fmu: Mean of the expectation. shape=[N,M]
         L  : Cholesky of covariance. shape=[N,N,M]
         Y  : Data. shape=[N',M']
        :return
         Stochastic approximation of
         \integ{logp(Y|f) N(f|Fmu,LLt) df}
        """
        # normal random vector with shape [M, N, num_stocastic_points]
        rndn = tf.random_normal(
                    [tf.shape(L)[2], tf.shape(L)[0], self.num_stocastic_points],
                    dtype=tf.float64)
        # Fampled point of F.
        # X.shape = [N, M, num_stocastic_points]. Mean: Fmu, Cov: LLt
        X = tf.tile(tf.expand_dims(Fmu,2), [1,1,self.num_stocastic_points]) + \
            tf.transpose(
                tf.batch_matmul(tf.transpose(L, [2,0,1]), rndn)
                # shape=[M, N, num_stocastic_points]
                , [1,0,2]) # shape [N,M,num_stocastic_points]
        # expand Y into the shape [N', M', num_stocastic_points]
        Y = tf.tile(tf.expand_dims(Y,2), [1, 1, self.num_stocastic_points])
        # logp.shape = [N", M", num_stocastic_points]
        batch_logp = self.batch_logp(X, Y)
        # weight matrix. Uniform weight. shape [N, M, num_stocastic_points, 1]
        weight = tf.ones([tf.shape(batch_logp)[0], self.num_stocastic_points,1],
                        dtype=tf.float64) / self.num_stocastic_points
        # return total of all the values and devide by num_stocastic_points.
        return tf.squeeze(tf.batch_matmul(batch_logp, weight))

    def logp(self, X, Y):
        """
        logp(Y|X)

        :args
         X: shape=[N,M]
         Y: shape=[N',M]
        :returns
         logp(Y|X): shape=[N',M]
        """
        raise NotImplementedError

    def batch_logp(self, X, Y):
        """
        logp(Y|X) for the batch calculation

        :args
         X: shape=[N,M,num_stocastic_points]
         Y: shape=[N',M,num_stocastic_points]
        :returns
         logp(Y|X): shape=[N',M, num_stocastic_points]
        """
        raise NotImplementedError


class Gaussian(StochasticLikelihood):
    """
    i.i.d Gaussian with uniform variance.
    Stochastic expectation is used.
    """
    def __init__(self, num_stocastic_points=20):
        StochasticLikelihood.__init__(self, num_stocastic_points)
        self.variance = Param(1.0, transforms.positive)

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)
    def batch_logp(self, F, Y):
        return self.logp(F, Y)

    def stochastic_expectations(self, Fmu, L, Y):
        return StochasticLikelihood.stochastic_expectations(self, Fmu, L, Y)

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
