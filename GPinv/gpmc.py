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
from GPflow.gpmc import GPMC
from GPflow.likelihoods import Likelihood
from GPflow.tf_wraps import eye
from .mean_functions import Zero
from .likelihoods import TransformedLikelihood
'''
class TransformedGPMC(GPMC):
    """
    The same with GPflow.gpmc.GPMC, but can accept TransformedLikelihood.
    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=Zero(), num_latent=None):
        # assert likelihood is an instance of TransformedLikelihood
        assert isinstance(likelihood, TransformedLikelihood)
        GPMC.__init__(self, X, Y, kern, likelihood, mean_function, num_latent)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.
            \log p(Y, V | theta).
        """
        L = self.getCholesky()
        F = tf.matmul(L, self.V) + self.mean_function(self.X)
        # TransformedLikelihood shoule have logp_gpmc method.
        return tf.reduce_sum(self.likelihood.logp_gpmc(F, self.Y))

    def getCholesky(self):
        K = self.kern.K(self.X)
        return tf.cholesky(K + eye(tf.shape(K)[0])*1e-6)
'''
