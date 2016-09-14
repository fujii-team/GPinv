import tensorflow as tf
import numpy as np
from .param import Param, DataHolder, Parameterized, ParamList, MinibatchData
from GPflow import transforms
from GPflow.tf_wraps import eye
from GPflow import svgp
from functools import reduce
from .mean_functions import Zero

class ModelInput(object):
    """
    An object used for constructing a multi-latent models.
    """
    def __init__(self, X, kern, Z, mean_function=Zero(),
                        X_minibatch_size=None, random_seed=0):
        """
        :param 2d-np.array X: Expressive coordinate
        :param Kern kern: GPinv.Kern object
        :param 2d-np.array Z: Inducing coordinate
        :param MeanFunction mean_functions: GPinv.MeanFunction object
        :param integer X_minibatch_size: size of X-minibatch. X-minibatching
            is not used, it should be None (default).
        :param integer random_seed: Random seed for X-minibatching.
        """
        self.X = X
        self.kern = kern
        self.Z = Z
        self.mean_function = mean_function
        self.X_minibatch_size = X_minibatch_size
        self.random_seed = random_seed

class IndexedDataHolder(DataHolder):
    """
    A DataHolder for the multiple-latent model input.
    We extend dimension of X where the last dimension shows the data index.
    """
    def __init__(self, input_list):
        """
        - input_list: List of ModelInput object.
        """
        DataHolder.__init__(self, np.zeros((1,1)))
        self.data_holders = []
        for data in input_list:
            index = 1.0*len(self.data_holders)
            # TODO add float_type support
            X = np.hstack([data.X, np.ones((data.X.shape[0],1))*index])
            if data.X_minibatch_size:
                self.data_holders.append(
                    MinibatchData(X, minibatch_size=data.X_minibatch_size,
                                rng=np.random.RandomState(data.random_seed)))
            else:
                self.data_holders.append(
                    DataHolder(X, on_shape_change='recompile'))

    def __getitem__(self, index):
        # Returns the data part of the array
        return self.data_holders[index].value[:,:-1]

    def __setitem__(self, index, item):
        # Set the data and attach the additional index
        self.data_holders[index].set_data(
            # TODO add float_type support
            np.hstack([item, np.ones((item.shape[0],1))*index]))

    def concat(self):
        """
        Returns the concat vector.
        """
        return np.vstack([d._array for d, i
            in zip(self.data_holders, list(range(len(self.data_holders))))])

    def get_feed_dict(self):
        # returns all the data
        return {self._tf_array: self.concat()}


class IndexedParamList(ParamList):
    """
    A list of Param, whose last dimension is the index.
    """
    def __init__(self, input_list):
        """
        - input_list: List of ModelInput object.
        """
        param_list = []
        index_list = []
        for data in input_list:
            param_list.append(Param(data.Z))
            # TODO add float_type support
            index_list.append(np.ones((data.Z.shape[0],1)) * len(index_list))
        ParamList.__init__(self, param_list)
        self.index_data = np.vstack(index_list)

    def concat(self):
        """
        Returns the concat vector with index.
        """
        if self._tf_mode:
            return tf.concat(1, [tf.concat(0, [z._tf_array for z in self._list]),
                                self.index_data])
        else: # returns np-array
            return np.hstack([np.vstack([z.value for z in self._list]),
                                self.index_data])

class ConcatParamList(ParamList):
    """
    A list of parameters that consists of multiple component and will be used
    as one large parameter.

    self.concat() provides the large np.array (or tf.tensor in _tf_mode)
    """
    def __init__(self, input_list, num_latent, value_list=None,
                                            transform=transforms.Identity()):
        """
        - input_list: List of ModelInput object.
        - num_latent: integer, indicating number of latent function M.
        - value_list: list of initial values
        """
        # assertion
        if value_list is not None:
            assert isinstance(value_list, list)
            assert len(value_list) == len(input_list)
        # number of all Z
        num_Z = np.sum([d.Z.shape[0] for d in input_list])
        # list of Param object
        param_list = []
        self.paddings_list = []
        num_Z_i = 0
        for i in range(len(input_list)):
            if value_list is None:
                param_list.append(Param(
                            np.zeros((input_list[i].Z.shape[0], num_latent)),
                            transform))
            else:
                param_list.append(Param(value_list[i], transform))
            self.paddings_list.append(
                        [[num_Z_i, num_Z - num_Z_i - input_list[i].Z.shape[0]], [0,0]])
            num_Z_i += input_list[i].Z.shape[0]
        # remember to call the parent's initializer
        ParamList.__init__(self, param_list)

    def concat(self):
        """
        Return q_sqrt matrix for whole the coordinates.
        """
        if self._tf_mode:
            return reduce(tf.add, [tf.pad(param._tf_array, paddings)
                                    for param, paddings
                                    in zip(self._list, self.paddings_list)])
        else: # return np
            return reduce(np.add, [np.pad(param.value, paddings, mode='constant')
                                    for param, paddings
                                    in zip(self._list, self.paddings_list)])

class SqrtParamList(ConcatParamList):
    """
    A list of Block matrix parameters, used for the Cholesky factor parameter.
    """
    def __init__(self, input_list, num_latent,
                    q_shape='fullrank', indices_list=None, q_sqrt_list=None):
        """
        - input_list: List of ModelInput object.
        - num_latent: integer, indicating number of latent function M.
        - q_shape is one of ['fullrank', 'specified']
                    if q_diag is 'specified', indices_list is necessary.
        - indices_list: List of tuples that indicates the non-zero matrix
                            blocks. Each tuple should be like (i, j)
                            with i >= j, where i-th and j-th parameters have
                            correlation, and N is the size of the corresponding
                            parameter.
        - q_sqrt_list: List of 3d-np.array with size [N,N,M].
                            They should indicate the initial values of q_sqrt.
                            If None, they are set to the default values.
        """
        assert q_shape in ['fullrank', 'specified']
        if q_shape is 'specified':
            assert isinstance(indices_list, list)

            if q_sqrt_list is not None:
                assert isinstance(q_sqrt_list, list)
                assert len(q_sqrt_list) == len(indices_list)

        # generate indices_list for the fullrank case
        if q_shape is 'fullrank':
            indices_list = []
            for i in range(len(input_list)):
                for j in range(i+1):
                    indices_list.append([i,j])

        param_list = []    # list of Param object
        self.paddings_list = []
        for i in range(len(indices_list)):
            indices = indices_list[i]
            if q_sqrt_list is None:
                # diagonal block
                if indices[0] == indices[1]:
                    # TODO add float_type support
                    q_sqrt = np.array([np.eye(input_list[indices[0]].Z.shape[0])
                           for _ in range(num_latent)]).swapaxes(0, 2)
                # non-diagonal block
                elif indices[0] >= indices[1]:
                    # TODO add float_type support
                    q_sqrt = np.zeros((input_list[indices[0]].Z.shape[0],
                                       input_list[indices[1]].Z.shape[0],
                                       num_latent))
                else:
                    raise ValueError(indices[0], ' > ', indices[1], ' is not satisfied.')
            else:
                q_sqrt = q_sqrt_list[i]
            param_list.append(Param(q_sqrt))

            # generating paddings.
            paddings = [[0,0],[0,0],[0,0]]
            for i in range(2):
                for j in range(indices[i]):
                    paddings[i][0] += input_list[j].Z.shape[0]
                for j in range(indices[i]+1, len(input_list)):
                    paddings[i][1] += input_list[j].Z.shape[0]
            self.paddings_list.append(paddings)
        # remember to call the parent's initializer
        ParamList.__init__(self, param_list)
