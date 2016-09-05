from GPflow import param
import tensorflow as tf
from GPflow import transforms
from GPflow.tf_hacks import eye
from functools import reduce

class Param(param.Param):
    pass

class DataHolder(param.DataHolder):
    pass

class Parameterized(param.Parameterized):
    pass

class ParamList(param.ParamList):
    pass

class DiagL(Parameterized):
    """
    Cholesky object for a diagonal covariance matrix.
    """
    def __init__(self, diag_entries):
        """
        :param 2D-np.array diag_entries: Diagonal values of the L matrix.
            If the shape of diag_entries is [N,M], the resultant tensor shape is
            [M, N, N], where i-th matrix at [i,N,N] are a diagonal.
        """
        Parameterized.__init__(self)
        self.diag = Param(diag_entries, transform=transforms.positive)

    def L(self):
        return tf.batch_matrix_diag(tf.transpose(self.diag))

    def Linv(self):
        return tf.batch_matrix_diag(tf.transpose(tf.inv(self.diag)))

class SemiDiag(Parameterized):
    """
    Object that stores sub-diagonal matrix parameter.
    """
    def __init__(self, head_index, values, shape, transform):
        """
        :param tuple head_index: Index pair [i0,j0] that represents the left-top
            corner of the diagonal
        :param 2D-np.array values: The values at the diagonal. Shape [N, M]
        :param tuple shape: The shape of the whole matrix.

        The resultant tensor shape is [M, N, N],
        where in each layer the [N,N]-matrix Lm stored.
        The matrix Lm[i,j] is sub-diagonal,

        Lm[i0+n,j0+n] = values[n] for n in [0,1,2,...,N-1]
        otherwise: zero.
        """
        Parameterized.__init__(self)
        # Diag entries are stored as Param
        self.diag = Param(values, transform)
        # paddings that is used to create the total matrix.
        self.paddings=[[0,0],
                       [head_index[0], shape[0]-values.shape[0]-head_index[0]],
                       [head_index[1], shape[1]-values.shape[0]-head_index[1]],]

    @property
    def matrix(self):
        mat = tf.batch_matrix_diag(tf.transpose(self.diag))
        return tf.pad(mat, self.paddings)


class MultiDiagL(Parameterized):
    """
    Cholesky object for diagonal-block covariance matrix.

    The object is written by

    L = sum_{i=1}{N}L^i(l,m,n)

    where L^i(l,m,n) is a matrix with the element
    if j-l==k-m and j-l<n and k-m<n:
        then L^i(l,m,n)[j,k] is non-zero
    otherwise: zero
    """
    def __init__(self, head_indices, values, trans, shape):
        """
        :param list of tuples head_indices:
        :param list of 2D-np.array values:
        :param list of GPflow.transforms trans:
        :param tuple of int shape: Final matrix shape. 3-element.
        """
        # Assertion
        assert len(shape)==3, "shpe should be 3-element list"

        Parameterized.__init__(self)
        # list of matrices that represent off-diagonal entries.
        self.matrices = ParamList([
            SemiDiag(head_index, value, shape, tran)
            for (head_index, value, tran)
            in zip(head_indices, values, trans)])
        self.shape = shape

    def L(self):
        return reduce(tf.add, [mat.matrix for mat in self.matrices])

    def Linv(self):
        I = tf.tile(tf.expand_dims(eye(self.shape[0]),0), [self.shape[2], 1, 1])
        return tf.batch_matrix_triangular_solve(
                    self.L(), I, lower=True, adjoint=None)

class DenseL(Parameterized):
    """
    # TODO
    Cholesky object for dense covariance matrix.
    """
    def __init__(self, diag_element, offdiag_element, shape):
        """
        :param 1D-np.array diag_element: diagonal element. Positive transform
            will be applied. Shape [N].
        :param 1D-np.array offdiag_element: offdiag_element. Offdiag transform
            will be applied. Shape [N*(N-1)//2]
        :param tuple of int shape: shape of L matrix.
        """
        assert shape[0] == shape[1]



'''

Currently tf.SparseTensor does not track the gradient.
I will wait the future improvement of TensorFlow.

class SparseParam(Parameterized):
    def __init__(self, indices, entries, shape, transform=transforms.Identity()):
        """
        Parameters that represents the sparse matrix.
        :param np.array indices: list of entries. 2D np.array that contains row
                                and column indices for the non-zero entries.
        :param np.array entries: 1D np.array that is the initial value for the
                                entries. indices.shape[0] == entries.shape[0]
        :param tuple shape: shape of the full-tensor. shape[0] == shape[1].

        """
        Parameterized.__init__(self)
        # assert indices and values have the same size
        assert(indices.shape[0] == entries.shape[0])

        # TODO add a sort operation so that
        # indices should be sorted in row-major order
        # (or equivalently lexicographic order on the tuples indices[i]).
        self.entries = Param(entries, transform=transform)
        self.indices = indices
        self.tensor_shape = shape

    @property
    def tensor(self):
        return tf.SparseTensor(self.indices, self.entries, self.tensor_shape)


class SparseL(Parameterized):
    """
    Object to parameterize the Cholesky lower triangular matrix L.
    """
    def __init__(self,   diag_indices,    diag_entries,
                      offdiag_indices, offdiag_entries, shape):
        Parameterized.__init__(self)
        # TODO add a check operation so that indices only contains the lower
        # half of the matrix

        self.diag    = SparseParam(  diag_indices,   diag_entries, shape,
                                        transform=transforms.positive)
        self.offdiag = SparseParam(offdiag_indices,offdiag_entries, shape)


    def L(self):
        return tf.sparse_tensor_to_dense(
                        tf.sparse_add(self.diag.tensor, self.offdiag.tensor),
                        default_value=0, validate_indices=False, name=None)


    def Linv(self):
        return tf.matrix_triangular_solve(self.L(), eye(self.diag.tensor_shape[0]),
                                            lower=True, adjoint=None)
'''
