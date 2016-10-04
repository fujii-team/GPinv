# This is a modification of GPflow/gpmc.py by Keisuke Fujii.
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
from GPflow import gpmc
from GPflow.likelihoods import Likelihood
from GPflow.tf_wraps import eye
from .mean_functions import Zero
from . import conditionals

class GPMC(gpmc.GPMC):
    """
    The same with GPflow.gpmc.GPMC, but can accept GPinv.kernels.Kern.
    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None):

        num_latent = num_latent or Y.shape[1]
        if mean_function is None:
            mean_function = Zero(num_latent)
        gpmc.GPMC.__init__(self, X, Y, kern, likelihood, mean_function, num_latent)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.
            \log p(Y, V | theta).
        """
        L = self.kern.Cholesky(self.X) # size [R,n,n]
        F = tf.transpose(tf.squeeze(   # size [n,R]
            tf.batch_matmul(
                tf.transpose(L,[2,0,1]), # [n,n,R] -> [R,n,n]
                tf.expand_dims(tf.transpose(self.V), -1)),[-1]))\
                                    + self.mean_function(self.X)
        # TransformedLikelihood shoule have logp_gpmc method.
        return tf.reduce_sum(self.likelihood.logp(tf.expand_dims(F,0), self.Y))

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.X, self.kern, self.V,
                                   q_sqrt=None, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Xnew), var
