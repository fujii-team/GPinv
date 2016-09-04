import tensorflow as tf
from GPflow.likelihoods import Likelihood
from GPflow import transforms
from GPflow.param import Param, DataHolder
from GPflow.likelihoods import Likelihood
from GPflow import densities

class CorrelatedLikelihood(Likelihood):
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
         Y  : Data. shape=[N,M]
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
        # expand Y into the shape [N, M, num_stocastic_points]
        Y = tf.tile(tf.expand_dims(Y,2), [1, 1, self.num_stocastic_points])
        # logp.shape = [N', M', num_stocastic_points]
        logp = self.logp(X, Y)
        # weight matrix. Uniform weight. shape [N, M, num_stocastic_points, 1]
        weight = tf.ones([tf.shape(logp)[0], self.num_stocastic_points,1],
                        dtype=tf.float64) / self.num_stocastic_points
        # return total of all the values and devide by num_stocastic_points.
        return tf.squeeze(tf.batch_matmul(logp, weight))

    def logp(self, X, Y):
        """
        logp(Y|X)

        :args
         X: shape=[N,M,num_stocastic_points]
         Y: shape=[N,M,num_stocastic_points]
        :returns
         logp(Y|X): shape=[N',M', num_stocastic_points]
        """
        raise NotImplementedError

class Gaussian(CorrelatedLikelihood):
    """
    i.i.d Gaussian with uniform variance.
    Stochastic expectation is used.
    """
    def __init__(self, num_stocastic_points=20):
        CorrelatedLikelihood.__init__(self, num_stocastic_points)
        self.variance = Param(1.0, transforms.positive)

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)

    def stochastic_expectations(self, Fmu, L, Y):
        # TODO should be changed to analytical expression.
        return CorrelatedLikelihood.stochastic_expectations(self, Fmu, L, Y)
