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
import numpy as np
from GPflow.priors import Gaussian
from .model import StVmodel
from .param import Param, DataHolder
from .mean_functions import Zero
from . import conditionals

class GPMC(StVmodel):
    """
    The same with GPflow.gpmc.GPMC, but can accept GPinv.kernels.Kern.
    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None):
        num_latent = num_latent or Y.shape[1]
        if mean_function is None:
            mean_function = Zero(num_latent)

        self.X = DataHolder(X, on_shape_change='recompile')
        self.Y = DataHolder(Y, on_shape_change='recompile')
        StVmodel.__init__(self, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.V = Param(np.zeros((self.num_latent,self.num_data)))
        self.V.prior = Gaussian(0., 1.)

    def _compile(self):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.
        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V = Param(np.zeros((self.num_latent,self.num_data)))
            self.V.prior = Gaussian(0., 1.)
        super(GPMC, self)._compile()

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.
            \log p(Y, V | theta).
        """
        f = self._sample(1) # [1,n,R]
        return tf.reduce_sum(self.likelihood.logp(f, self.Y))

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.X, self.kern, self.V,
                                    q_sqrt=None, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Xnew), var

    def _sample(self, n_sample):
        """
        Calculate GP function f from the current latent variables V and
        hyperparameters.
        """
        L = self.kern.Cholesky(self.X) # size [R,n,n]
        F = tf.transpose(tf.squeeze(   # size [n,R]
            tf.batch_matmul(L,tf.expand_dims(self.V, -1)),[-1]))\
                                    + self.mean_function(self.X)
        return tf.tile(tf.expand_dims(F, 0), [n_sample,1,1]) # size [N,n,R]
