import tensorflow as tf
import numpy as np
from .param import ConcatDataHolder, ConcatParamList, SqrtParamList
from .kernels import BlockDiagonal
from .mean_functions import Zero, SwitchedMeanFunction
from . import transforms

class ModelInput(object):
    """
    An object used for constructing a multi-latent models.
    """
    def __init__(self, X, kern, Z=None, mean_function=Zero(),
                        X_minibatch_size=None, random_seed=0,
                        q_mu = None, q_sqrt = None, num_latent=1):
        """
        :param 2d-np.array X: Expressive coordinate
        :param Kern kern: GPinv.Kern object
        :param 2d-nkp.array Z: Inducing coordinate
        :param MeanFunction mean_functions: GPinv.MeanFunction object
        :param integer X_minibatch_size: size of X-minibatch. X-minibatching
            is not used, it should be None (default).
        :param integer random_seed: Random seed for X-minibatching.
        :param q_mu: 2d-np.array.
        """
        self.X = X
        self.kern = kern
        if Z is None:
            self.Z = X.copy()
        else:
            self.Z = Z
        self.mean_function = mean_function
        self.X_minibatch_size = X_minibatch_size or X.shape[0]
        self.random_seed = random_seed
        self.q_mu = q_mu
        if self.q_mu is None:         # Setting default values
            self.q_mu = np.zeros((self.Z.shape[0], num_latent))
        self.q_sqrt = q_sqrt


class ModelInputSet(object):
    """
    An object to handle the multiple ModelInput
    """
    def __init__(self, input_list, num_latent = 1,
        q_shape = None,
        q_indices_list = None, q_sqrt_list=None, jitter=0.
    ):
        """
        - input_list: List of the ModelInput.
        - num_latent: number of latent functions that share kernel.
        - q_shape : one of ['None', 'fullrank', 'block_diagonal','diagonal',
                    'specified'], which are used for the svgp model.
                    If None, then this class is assumed to not be used in svgp.
        - q_indices_list: List of tuples that indicates the non-zero matrix in
                    SqrtParamList.
                    Each tuple should be like (i, j) with i >= j, where i-th and
                    j-th parameters have correlation, and N is the size of the
                    corresponding parameter.
        - q_sqrt_list: List of 3d-np.array with size [N,N,M].
                    They should indicate the initial values of q_sqrt.
                    If None, they are set to the default values.
        """
        self.input_list = input_list
        self.num_latent = num_latent
        self.q_shape = q_shape
        self.jitter=jitter
        # --- set the default values ---
        if q_shape is not None:
            assert q_shape in ['fullrank', 'block_diagonal','diagonal','specified']
            if q_indices_list is None:
                q_indices_list = []
                # generate indices_list for the fullrank case
                if q_shape is 'fullrank':
                    for i in range(len(input_list)):
                        for j in range(i+1):
                            q_indices_list.append([i,j])
                # generate indices_list for the block_diagonal case
                elif q_shape is 'block_diagonal':
                    for i in range(len(input_list)):
                        q_indices_list.append([i,i])
                elif q_shape is 'specified':
                    assert q_indices_list is not None

            self.q_indices_list = q_indices_list
            if q_sqrt_list is None:
                self.q_sqrt_list = self.generateDefault_qsqrt()
            else:
                self.q_sqrt_list = [q_sqrt.copy() for q_sqrt in q_sqrt_list]
            # generate paddings for q_sqrt
            self.q_paddings_list = self.generateQsqrtPaddings()


    def getConcat_X(self):
        """
        Return the set of X as ConcatDataHolder
        """
        return ConcatDataHolder(
            [d.X for d in self.input_list], # Xlist
            [d.X_minibatch_size for d in self.input_list],
            [d.random_seed for d in self.input_list]
        )

    def getConcat_Z(self):
        """
        Return the set of Z as ConcatParamList
        """
        return ConcatParamList([d.Z for d in self.input_list])

    def getConcat_q_mu(self):
        """
        Return the set of q_mu as ConcatParamList
        """
        return ConcatParamList([d.q_mu for d in self.input_list])

    def getConcat_q_sqrt(self):
        """
        Return the set of q_sqrt as
        """
        if self.q_shape is 'diagonal':
            return ConcatParamList(self.q_sqrt_list, transforms.positive)
        else:
            return SqrtParamList(self.q_indices_list, self.q_sqrt_list,
                                                      self.q_paddings_list)

    def generate_X_slices(self):
        """
        make slice_X_begin and slice_X_size
        """
        # slice index used for the K(X) computing
        slice_X_begin,  slice_X_size  = [], []
        num_X_i = 0
        for d in self.input_list:
            slice_X_begin.append(num_X_i)
            slice_X_size.append(d.X_minibatch_size)
            num_X_i += d.X_minibatch_size
        return slice_X_begin, slice_X_size

    def generate_Z_slices(self):
        """
        make slice_X_begin and slice_X_size
        """
        # slice index used for the K(X) computing
        slice_X_begin,  slice_X_size  = [], []
        num_X_i = 0
        for d in self.input_list:
            slice_X_begin.append(num_X_i)
            slice_X_size.append(d.Z.shape[0])
            num_X_i += d.Z.shape[0]
        return slice_X_begin, slice_X_size


    def getKernel(self, jitter=None):
        # slice index used for the K(X,X2) computing
        if jitter is None:
            jitter = self.jitter
        slice_X, size_X = self.generate_Z_slices()
        slice_X2,size_X2= self.generate_X_slices()
        kern = BlockDiagonal([d.kern for d in self.input_list],
                                slice_X,  size_X,
                                slice_X2, size_X2, jitter)
        return kern

    def getMeanFunction(self):
        # slice index used for the K(X) computing
        slice_X,size_X= self.generate_X_slices()
        return SwitchedMeanFunction([d.mean_function for d in self.input_list],
                            slice_X, size_X)

    def generateDefault_qsqrt(self):
        """
        Set the default value if they are not assigned explicitly.
        """
        q_sqrt_list = []
        if self.q_shape in ['fullrank', 'block_diagonal', 'specified']:
            for indices in self.q_indices_list:
                # diagonal block
                if indices[0] == indices[1]:
                    # TODO add float_type support
                    q_sqrt = np.array([np.eye(self.input_list[indices[0]].Z.shape[0])
                           for _ in range(self.num_latent)]).swapaxes(0, 2)
                # non-diagonal block
                elif indices[0] >= indices[1]:
                    # TODO add float_type support
                    q_sqrt = np.zeros((self.input_list[indices[0]].Z.shape[0],
                                       self.input_list[indices[1]].Z.shape[0],
                                       self.num_latent))
                else:
                    raise Exception('Bad q_indice_list')
                q_sqrt_list.append(q_sqrt)
            return q_sqrt_list

        elif self.q_shape is 'diagonal':
            for d in self.input_list:
                q_sqrt_list.append(np.ones((d.Z.shape[0], self.num_latent)))
            return q_sqrt_list

        return None

    def generateQsqrtPaddings(self):
        # generating paddings.
        paddings_list = []
        if self.q_shape is not 'diagonal':
            for indices in self.q_indices_list:
                paddings = [[0,0],[0,0],[0,0]]
                for i in range(2):
                    for j in range(indices[i]):
                        paddings[i][0] += self.input_list[j].Z.shape[0]
                    for j in range(indices[i]+1, len(self.input_list)):
                        paddings[i][1] += self.input_list[j].Z.shape[0]
                paddings_list.append(paddings)

        return paddings_list
