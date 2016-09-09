import tensorflow as tf
from GPinv import transforms
from GPinv.param import SemiDiag, MultiDiagL
import numpy as np
import unittest

class test_semidiag(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        rng = np.random.RandomState(0)
        N = 5
        n = 3
        shape = (N, N)
        values = rng.randn(n,2)
        head_index = (1,0)
        semiDiag = SemiDiag(head_index, values, shape, transforms.Identity())

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        free_var = semiDiag.get_free_state()
        semiDiag.make_tf_array(free_var)

        with semiDiag.tf_mode():
            semiDiag_np = sess.run(semiDiag.matrix)

        semiDiag_ref = np.zeros((2,N,N))
        for k in range(2):
            for i in range(3):
                semiDiag_ref[k, i+1, i+0] = values[i,k]

        self.assertTrue(np.allclose(semiDiag_ref, semiDiag_np))


class test_MultiDiagL(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        rng = np.random.RandomState(0)
        N = 5
        n1 = 5
        n2 = 3
        shape = (N, N, 2)
        values = [np.exp(rng.randn(n1,2)), rng.randn(n2,2)]
        head_indices = [(0,0), (1,0)]
        trans = [transforms.positive, transforms.Identity()]
        multiDiagL = MultiDiagL(head_indices, values, trans, shape)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        free_var = multiDiagL.get_free_state()
        multiDiagL.make_tf_array(free_var)

        with multiDiagL.tf_mode():
            multiDiagL_np = sess.run(multiDiagL.L())
            multiDiagLinv_np = sess.run(multiDiagL.Linv())

        multiDiagL_ref = np.zeros((2,N,N))
        for k in range(2):
            for i in range(n1):
                multiDiagL_ref[k, i+0, i+0] = values[0][i,k]
            for i in range(n2):
                multiDiagL_ref[k, i+1, i+0] = values[1][i,k]

        self.assertTrue(np.allclose(multiDiagL_ref, multiDiagL_np))

        # make sure that L[i] @ Linv[i] = Identity for each layer
        for k in range(2):
            self.assertTrue(np.allclose(
                np.eye(N, N),
                np.dot(multiDiagL_np[k,:,:], multiDiagLinv_np[k,:,:])))
'''
class test_sparse_param(unittest.TestCase):
    def setUp(self):
        pass

    def test_L(self):
        offdiag_indices = np.array([
            [1,0],
            [2,0],
            [3,0], [3,1]
        ], dtype=np.int64)
        diag_indices = np.array([[0,0], [1,1], [2,2], [3,3]])

        rng = np.random.RandomState(0)
        diag_entries = np.exp(rng.randn(diag_indices.shape[0]))
        offdiag_entries = rng.randn(offdiag_indices.shape[0])
        shape = [4,4]

        sparseL = SparseL(diag_indices, diag_entries,
                           offdiag_indices, offdiag_entries, shape)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf_array = sparseL.get_free_state()
        sparseL.make_tf_array(tf_array)
        with sparseL.tf_mode():
            L = sess.run(sparseL.L())
            Linv = sess.run(sparseL.Linv())

        for index, entry in zip(diag_indices, diag_entries):
            self.assertTrue(np.allclose(L[index[0],index[1]], entry))

        for index, entry in zip(offdiag_indices, offdiag_entries):
            self.assertTrue(np.allclose(L[index[0],index[1]], entry))

        # assert L * L^T = I
        I = np.dot(L, Linv)
        self.assertTrue(np.allclose(I, np.eye(4)))
'''

if __name__ == "__main__":
    unittest.main()
