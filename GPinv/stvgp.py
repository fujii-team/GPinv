# This is a modification of GPflow/vgp.py by Keisuke Fujii.
#
# The original source file is distributed at
# https://github.com/GPflow/GPflow/blob/master/GPflow/svgp.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from GPflow.densities import gaussian
from GPflow.model import GPModel
from GPflow import transforms,kullback_leiblers
from GPflow.param import AutoFlow
from GPflow.tf_wraps import eye
from GPflow._settings import settings
from .mean_functions import Zero
from .param import Param, DataHolder
from . import conditionals
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class StVGP(GPModel):
    """
    Stochastic approximation of the Variational Gaussian process
    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None,
                 q_shape='fullrank',
                 KL_analytic=False,
                 num_samples=20):
        """
        X is a data matrix, size n x D
        Y is a data matrix, size n x R
        kern, likelihood, mean_function are appropriate GPflow objects
        q_shape: 'fullrank', 'diagonal', or integer less than n.
        KL_analytic: True for the use of the analytical expression for KL.
        num_samples: number of samples to approximate the posterior.
        """
        self.num_data = X.shape[0] # number of data, n
        self.num_latent = num_latent or Y.shape[1] # number of latent function, R
        self.num_samples = num_samples # number of samples to approximate integration, N
        if mean_function is None:
            mean_function = Zero(self.num_latent)
        # if minibatch_size is not None, Y is stored as MinibatchData.
        # Note that X is treated as DataHolder.
        Y = DataHolder(Y, on_shape_change='recompile')
        X = DataHolder(X, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        # variational parameter.
        # Mean of the posterior
        self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
        # If true, mean-field approimation is made.
        self.q_shape = q_shape
        # Sqrt of the covariance of the posterior
        # diagonal
        if self.q_shape == 'diagonal':
            self._q_sqrt = Param(np.ones((self.num_data, self.num_latent)),
                                transforms.positive)
        # fullrank
        elif self.q_shape == 'fullrank':
            q_sqrt = np.array([np.eye(self.num_data)
                                for _ in range(self.num_latent)]).swapaxes(0, 2)
            self._q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform                              transforms.positive)
        # multi-diagonal-case
        elif isinstance(self.q_shape, int):
            # make sure q_shape is within 1 < num_data
            assert(self.q_shape > 1 and self.q_shape < self.num_data)
            q_sqrt = np.zeros((self.num_data, self.q_shape, self.num_latent),
                                np_float_type)
            # fill one in diagonal value
            q_sqrt[:,0,:] = np.ones((self.num_data, self.num_latent),np_float_type)
            self._q_sqrt = Param(q_sqrt)
        self.KL_analytic = KL_analytic

    def _compile(self, optimizer=None, **kw):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            raise NotImplementedError
            '''
            self.num_data = self.X.shape[0]
            self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
            if self.q_diag:
                self.q_sqrt = Param(np.ones((self.num_data, self.num_latent)),
                                    transforms.positive)
            else:
                q_sqrt = np.array([np.eye(self.num_data)
                                    for _ in range(self.num_latent)]).swapaxes(0, 2)
                self.q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform                              transforms.positive)
            '''
        return super(StVGP, self)._compile(optimizer=optimizer, **kw)

    def build_likelihood(self):
        """
        This method computes the variational lower bound on the likelihood, with
        stochastic approximation.
        """
        f_samples = self._sample(self.num_samples)
        # In likelihood, dimensions of f_samples and self.Y must be matched.
        lik = tf.reduce_sum(self.likelihood.logp(f_samples, self.Y))
        if not self.KL_analytic:
            return (lik - self._KL)/self.num_samples
        else:
            return lik/self.num_samples - self._analytical_KL()

    def build_predict(self, Xnew, full_cov=False):
        """
        Prediction of the latent functions.
        The posterior is approximated by multivariate Gaussian distribution.

        :param tf.tensor Xnew: Coordinate where the prediction should be made.
        :param bool full_cov: True for return full covariance.
        :return tf.tensor mean: The posterior mean sized [n,R]
        :return tf.tensor var: The posterior mean sized [n,R] for full_cov=False
                                                      [n,n,R] for full_cov=True.
        """
        mu, var = conditionals.conditional(Xnew, self.X, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Xnew), var

    @AutoFlow((tf.int32, []))
    def sample_F(self, n_sample):
        """
        Get samples of the latent function values at the observation points.
        :param integer n_sample: number of samples.
        :return tf.tensor: Samples sized [N,n,R]
        """
        f_samples = self._sample(n_sample[0])
        return self.likelihood.sample_F(f_samples)

    @AutoFlow((tf.int32, []))
    def sample_Y(self, n_sample):
        """
        Get samples of the latent function values at the observation points.
        :param integer n_sample: number of samples.
        :return tf.tensor: Samples sized [N,n,R]
        """
        f_samples = self._sample(n_sample[0])
        return self.likelihood.sample_Y(f_samples)

    @property
    def q_sqrt(self):
        """
        Reshape self._q_sqrt param to [R,n,n]
        """
        # Match dimension of the posterior variance to the data.
        # diagonal case
        if self.q_shape == 'diagonal':
            return tf.transpose(
                    tf.batch_matrix_diag(tf.transpose(self._q_sqrt)), [1,2,0])
        else:
            if self.q_shape == 'fullrank':
                sqrt = self._q_sqrt
            # multi-diagonal-case
            else:
                n,R,q = self.num_data, self.num_latent, self.q_shape
                # shape [R, n, q]
                paddings = [[0,0],[0, n-q+1],[0,0]]
                # shape [R, n, n+1]
                sqrt = tf.transpose(tf.reshape(tf.slice(tf.reshape(
                                tf.pad(self._q_sqrt, paddings),  # [n,n+1,R]
                                [n*(n+1),R]), [0,0], [n*n,R]), [n,n,R]),[1,0,2])
                             # [n*(n+1),R] -> [n*n,R] -> [n,n,R]
            # return with [n,n,R]
            return tf.transpose(tf.batch_matrix_band_part(
                                tf.transpose(sqrt,[2,0,1]), -1, 0), [1,2,0])

    def _sample(self, N):
        """
        :param integer N: number of samples
        :Returns
         samples picked from the variational posterior.
         The Kulback_leibler divergence is stored as self._KL
        """
        n = self.num_data
        R = self.num_latent
        sqrt = tf.transpose(self.q_sqrt, [2,0,1])
        # Log determinant of matrix S = q_sqrt * q_sqrt^T
        logdet_S = tf.cast(N, float_type)*tf.reduce_sum(
                tf.log(tf.square(tf.batch_matrix_diag_part(sqrt))))
        sqrt = tf.tile(tf.expand_dims(sqrt, 1), [1,N,1,1]) # [R,N,n,n]
        # noraml random samples, [R,N,n,1]
        v_samples = tf.random_normal([R,N,n,1], dtype=float_type)
        # Match dimension of the posterior mean, [R,N,n,1]
        mu = tf.tile(tf.expand_dims(tf.expand_dims(
                                tf.transpose(self.q_mu), 1), -1), [1,N,1,1])
        u_samples = mu + tf.batch_matmul(sqrt, v_samples)
        # Stochastic approximation of the Kulback_leibler KL[q(f)||p(f)]
        self._KL = - 0.5 * logdet_S\
             - 0.5 * tf.reduce_sum(tf.square(v_samples)) \
             + 0.5 * tf.reduce_sum(tf.square(u_samples))
        # Cholesky factor of kernel [R,N,n,n]
        L = tf.tile(tf.expand_dims(
                tf.transpose(self.kern.Cholesky(self.X), [2,0,1]),1), [1,N,1,1])
        # mean, sized [N,n,R]
        mean = tf.tile(tf.expand_dims(
                    self.mean_function(self.X),
                0), [N,1,1])
        # sample from posterior, [N,n,R]
        f_samples = tf.transpose(
                tf.squeeze(tf.batch_matmul(L, u_samples),[-1]), # [R,N,n]
                [1,2,0]) + mean
        # return as Dict to deal with
        return f_samples

    def _analytical_KL(self):
        """
        Analytically evaluate KL
        """
        if self.q_shape == 'diagonal':
            KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu, self._q_sqrt)
        else:
            KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt)
        return KL
