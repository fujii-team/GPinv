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
from GPflow.tf_wraps import eye
from GPflow.model import GPModel
from . import transforms
from .param import DataHolder, Param, Parameterized, ParamList, MinibatchData
from .kernels import BlockDiagonal
from .mean_functions import Zero, SwitchedMeanFunction
from .svgp import TransformedSVGP
from .multilatent_likelihoods import MultilatentLikelihood
from .multilatent_conditionals import conditional

class MultilatentSVGP(TransformedSVGP):
    """
    SVGP for the transformed likelihood with multiple latent functions.
    """
    def __init__(self, model_input_set,
                 Y, likelihood,
                 minibatch_size=None, random_seed=0):
        """
        - model_inputs: list of ModelInput objects.
        - Y is a data matrix, size N' x R
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_shape is one of ['fullrank', 'diagonal', 'specified']
        - q_indices_list is list of tuples, which indicates the corelation
                                                between each model_input.
        - minibatch_size is the size for the minibatching for Y
        - random_seed is the seed for the Y-minibatching.
        """
        self.model_input_set = model_input_set
        self.num_data = Y.shape[0]
        # currently, whiten option is not supported.
        self.whiten = True

        self.num_latent = self.model_input_set.num_latent

        # Construct input vector, kernel, and mean_functions from input_list
        X = self.model_input_set.getConcat_X()
        # make Y Data holder
        if minibatch_size is None:
            minibatch_size = self.num_data
            Y = DataHolder(Y)
        else:
            Y = MinibatchData(Y, minibatch_size, rng=np.random.RandomState(random_seed))

        kern          = self.model_input_set.getKernel()
        mean_function = self.model_input_set.getMeanFunction()

        # assert likelihood is appropriate
        assert isinstance(likelihood, MultilatentLikelihood)
        slice_begin, slice_end = self.model_input_set.generate_X_slices()
        likelihood.make_slice_indices(slice_begin, slice_end)

        self.Z = self.model_input_set.getConcat_Z()
        self.num_inducing = self.Z.shape[0]

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)

        # init variational parameters
        self.q_mu_list = self.model_input_set.getConcat_q_mu()
        self.q_sqrt_list = self.model_input_set.getConcat_q_sqrt()
        self.q_diag = True if self.model_input_set.q_shape is 'diagonal' else False

    @property
    def q_mu(self):
        return self.q_mu_list.concat()
    @property
    def q_sqrt(self):
        return self.q_sqrt_list.concat()

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.Z, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=self.whiten)
        return mu + self.mean_function(Xnew), var
