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
from GPflow import transforms
from GPflow.tf_wraps import eye
from GPflow._settings import settings
from .mean_functions import Zero
from .param import Param, DataHolder, MinibatchData
from . import conditionals

class StVGP(GPModel):
    """
    Stochastic approximation of the Variational Gaussian process
    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None,
                 minibatch_size=None,
                 q_diag=False,
                 num_samples=20):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects
        minibatch_size: if not None, minibatching is enabled ONLY for Y.
        q_diag: True for diagonal approximation of q.
        num_samples: number of samples to approximate the posterior.
        """
        self.num_data = X.shape[0] # number of data, n
        self.num_latent = num_latent or Y.shape[1] # number of latent function, R
        self.num_samples = num_samples # number of samples to approximate integration, N
        if mean_function is None:
            mean_function = Zero(self.num_latent)
        # if minibatch_size is not None, Y is stored as MinibatchData.
        # Note that X is treated as DataHolder.
        if minibatch_size is not None:
            Y = MinibatchData(Y)
        else:
            Y = DataHolder(Y, on_shape_change='recompile')
        X = DataHolder(X, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        # variational parameter.
        # Mean of the posterior
        self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
        # If true, mean-field approimation is made.
        self.q_diag = q_diag
        # Sqrt of the covariance of the posterior
        if self.q_diag:
            self.q_sqrt = Param(np.ones((self.num_data, self.num_latent)),
                                transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_data)
                                for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform                              transforms.positive)


    def _compile(self, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
            if self.q_diag:
                self.q_sqrt = Param(np.ones((self.num_data, self.num_latent)),
                                    transforms.positive)
            else:
                q_sqrt = np.array([np.eye(self.num_data)
                                    for _ in range(self.num_latent)]).swapaxes(0, 2)
                self.q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform                              transforms.positive)
        return super(StVGP, self)._compile(optimizer=optimizer)

    def build_likelihood(self):
        """
        This method computes the variational lower bound on the likelihood, with
        stochastic approximation.
        """
        N = self.num_samples
        n = self.num_data
        R = self.num_latent
        # Match dimension of the posterior variance to the data.
        if self.q_diag:
            sqrt = tf.batch_matrix_diag(tf.transpose(self.q_sqrt)) # [R,n,n]
        else:
            sqrt = tf.batch_matrix_band_part(
                            tf.transpose(self.q_sqrt,[2,0,1]), -1, 0) # [R,n,n]
        # Log determinant of matrix S = q_sqrt * q_sqrt^T
        logdet_S = 2.0*N*tf.reduce_sum(
                tf.log(tf.abs(tf.batch_matrix_diag_part(sqrt))))
        sqrt = tf.tile(tf.expand_dims(sqrt, 1), [1,N,1,1]) # [R,N,n,n]
        # noraml random samples, [R,N,n,1]
        v_samples = tf.random_normal([R,N,n,1], dtype=tf.float64)
        # Match dimension of the posterior mean, [R,N,n,1]
        mu = tf.tile(tf.expand_dims(tf.expand_dims(
                                tf.transpose(self.q_mu), 1), -1), [1,N,1,1])
        u_samples = mu + tf.batch_matmul(sqrt, v_samples)
        # Stochastic approximation of the Kulback_leibler KL[q(f)||p(f)]
        KL = - 0.5 * logdet_S\
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
        # In likelihood, dimensions of f_samples and self.Y must be matched.
        lik = tf.reduce_sum(self.likelihood.logp(f_samples, self.Y))
        return (lik - KL)/self.num_samples


    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.X, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Xnew), var
