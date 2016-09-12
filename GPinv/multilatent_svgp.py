# This is a modification of GPflow/svgp.py by Keisuke Fujii.
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

import tensorflow as tf
import numpy as np
from GPflow.svgp import SVGP, MinibatchData
from GPflow.model import GPModel
from GPflow.tf_wraps import eye
from .param import DataHolder, Param
from .mean_functions import Zero
from .model_input import ModelInput

class MultilatentSVGP(SVGP):
    """
    SVGP for the transformed likelihood with multiple latent functions.
    """
    def __init__(self, input_list,
                 Y, likelihood, num_latent=None, q_diag=False, whiten=True,
                 minibatch_size=None):
        """
        - model_inputs: list of ModelInput objects.
        - Y is a data matrix, size N' x R
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        """
        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]

        if X_minibatch:
            X = MinibatchData(X, minibatch_size)
        else:
            X = DataHolder(X)
        Y = MinibatchData(Y, minibatch_size)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Param(Z)
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = Z.shape[0]

        # init variational parameters
        self.q_mu = Param(np.zeros((self.num_inducing, self.num_latent)))
        if self.q_diag:
            self.q_sqrt = Param(np.ones((self.num_inducing, self.num_latent)),
                                transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform)


    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        # Get prior KL.
        KL = self.build_prior_KL()
        # Get conditionals
        fmean, fcov = self.build_predict(self.X, full_cov=True)
        # TODO Rank-two downgrade should be applied (if possible).
        jitter = tf.tile(tf.expand_dims(eye(tf.shape(self.X)[0]), [0]),
                        [self.num_latent, 1,1]) * 1.0e-6
        Lcov = tf.transpose(
                    tf.batch_cholesky(tf.transpose(fcov, [2,0,1]) + jitter), [1,2,0])
        # Get variational expectations.
        var_exp = self.likelihood.stochastic_expectations(fmean, Lcov, self.Y)
        # re-scale for minibatch size
        scale = tf.cast(self.num_data, tf.float64) / tf.cast(tf.shape(self.Y)[0], tf.float64)
        return tf.reduce_sum(var_exp) * scale - KL
