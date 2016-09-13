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
from .param import DataHolder, Param, Parameterized, ParamList, MinibatchData
from .multilatent_param import IndexedDataHolder, IndexedParamList, ConcatParamList, SqrtParamList
from .kernels import SwitchedKernel
from .mean_functions import Zero, SwitchedMeanFunction
from .svgp import TransformedSVGP


class MultilatentSVGP(TransformedSVGP):
    """
    SVGP for the transformed likelihood with multiple latent functions.
    """
    def __init__(self, input_list,
                 Y, likelihood, num_latent=None,
                 q_shape='fullrank',
                 q_indices_list=None,
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
        self.input_list = input_list
        # minibatch_size
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = Y.shape[0]

        if q_shape is 'diagonal':
            self.q_diag = True
        else:
            self.q_diag = False
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = np.sum([d.Z.shape[0] for d in self.input_list])

        # Construct input vector, kernel, and mean_functions from input_list
        X = IndexedDataHolder(self.input_list)
        Y = MinibatchData(Y, minibatch_size, rng=np.random.RandomState(random_seed))

        self.Z_list = IndexedParamList(self.input_list)
        kern          = SwitchedKernel([d.kern          for d in input_list])
        mean_function = SwitchedKernel([d.mean_function for d in input_list])
        # assert likelihood is appropriate
        assert isinstance(likelihood, MultilatentLikelihood)
        likelihood.make_slice_indices(self.input_list)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)

        # init variational parameters
        self.q_mu_list = ConcatParamList(self.input_list, self.num_latent)

        if self.q_diag:
            self.q_sqrt_list = ConcatParamList(self.input_list, self.num_latent,
                    [np.ones((z.shape[0], self.num_latent)) for z in self.Z_list],
                    transforms.positive)
        else:
            self.q_sqrt_list = SqrtParamList(self.input_list, self.num_latent,
                                                        q_shape, q_indices_list)

    @property
    def Z(self):
        return self.Z_list.concat()
    @property
    def q_mu(self):
        return self.q_mu_list.concat()
    @property
    def q_sqrt(self):
        return self.q_sqrt_list.concat()
