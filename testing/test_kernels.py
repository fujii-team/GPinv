from __future__ import print_function
import GPinv
from GPinv.kernels import RBF
import GPflow
import numpy as np
import unittest
import tensorflow as tf

class RefRBF(object):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def __init__(self, kern):
        self.kern = kern
        self.R = self.kern.variance.value.shape[0]

    def K(self, X, X2=None):
        """
        X and X2 is np.array
        """
        if X2 is None:
            X2 = X

        core = np.zeros((X.shape[0], X2.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X2.shape[0]):
                length = self.kern.lengthscales.value
                x_dif = (X[i] - X2[j])/length
                sq_dist = np.sum(x_dif*x_dif)
                core[i, j] = np.exp(-0.5 * sq_dist)
        K_mat = np.zeros((X.shape[0],X2.shape[0],self.R))
        for k in range(self.R):
            K_mat[:,:,k] = core[:,:] * self.kern.variance.value[k]
        return K_mat

    def Kdiag(self, X):
        K_diag = np.zeros((X.shape[0], self.kern.variance.value.shape[0]))
        for i in range(X.shape[0]):
            var = self.kern.variance.value
            K_diag[i] = var
        return K_diag


class test_rbf(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        var = np.exp(rng.randn(3))
        self.m = GPflow.param.Parameterized()
        self.m.kern = RBF(2, 3, var, ARD=True)
        self.X  = rng.randn(10,2)
        self.X2 = rng.randn(11,2)
        # prepare Parameterized
        tf_array = self.m.get_free_state()
        self.m.make_tf_array(tf_array)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def test_Kxx(self):
        with self.m.tf_mode():
            Kxx  = self.sess.run(self.m.kern.K(self.X))
        ref_rbf = RefRBF(self.m.kern)
        Kxx_ref = ref_rbf.K(self.X)
        self.assertTrue(np.allclose(Kxx, Kxx_ref))

    def test_Kxx2(self):
        with self.m.tf_mode():
            Kxx2 = self.sess.run(self.m.kern.K(self.X,self.X2))
        ref_rbf = RefRBF(self.m.kern)
        Kxx2_ref = ref_rbf.K(self.X,self.X2)
        self.assertTrue(np.allclose(Kxx2, Kxx2_ref))

    def test_Kdiag(self):
        with self.m.tf_mode():
            Kdiag = self.sess.run(self.m.kern.Kdiag(self.X))
        ref_rbf = RefRBF(self.m.kern)
        Kdiag_ref = ref_rbf.Kdiag(self.X)
        self.assertTrue(np.allclose(Kdiag, Kdiag_ref))

    def test_Cholesky(self):
        with self.m.tf_mode():
            K = self.sess.run(self.m.kern.K(self.X))
            chol = self.sess.run(self.m.kern.Cholesky(self.X))
        for i in range(chol.shape[2]):
            self.assertTrue(np.allclose(K[:,:,i],
                        np.dot(chol[:,:,i], np.transpose(chol[:,:,i]))))
'''
class test_RBF_csym(unittest.TestCase):
    """
    Test for the cylindrically-symmetric RBF kernel
    """
    def test(self):
        rng = np.random.RandomState(0)
        # setting
        X = rng.randn(10,2)
        X2= rng.randn(12,2)
        var = np.exp(rng.randn(1))
        lengthscales = np.exp(rng.randn(2))
        # constructing GPflow models
        m = GPflow.model.Model()
        m.kern = RBF_csym(2, ARD=True)
        m.kern.variance = var
        m.kern.lengthscales = lengthscales

        m.X = GPflow.param.DataHolder(X)
        m.X2= GPflow.param.DataHolder(X2)

        tf_array = m.get_free_state()
        m.make_tf_array(tf_array)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        with m.tf_mode():
            Kxx  = sess.run(m.kern.K(m.X), feed_dict = m.get_feed_dict())
            Kxx2 = sess.run(m.kern.K(m.X, m.X2), feed_dict = m.get_feed_dict())
            Kdiag= sess.run(m.kern.Kdiag(m.X), feed_dict = m.get_feed_dict())

        # reference
        Kxx_ref = np.zeros((X.shape[0],X.shape[0]))
        Kxx2_ref = np.zeros((X.shape[0],X2.shape[0]))
        Kdiag_ref = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                square_dist = 0
                square_dist2 = 0
                for k in range(X.shape[1]):
                    square_dist += ((X[i,k] - X[j,k])/lengthscales[k])**2.
                    square_dist2 += ((X[i,k] + X[j,k])/lengthscales[k])**2.
                Kxx_ref[i,j] = var * np.exp(-0.5*square_dist) + \
                               var * np.exp(-0.5*square_dist2)

            for j in range(X2.shape[0]):
                square_dist = 0
                square_dist2 = 0
                for k in range(X.shape[1]):
                    square_dist += ((X[i,k] - X2[j,k])/lengthscales[k])**2.
                    square_dist2 += ((X[i,k] + X2[j,k])/lengthscales[k])**2.
                Kxx2_ref[i,j] = var * np.exp(-0.5*square_dist) + \
                                var * np.exp(-0.5*square_dist2)

            square_dist = 0
            square_dist2 = 0
            for k in range(X.shape[1]):
                square_dist += ((X[i,k] - X[i,k])/lengthscales[k])**2.
                square_dist2 += ((X[i,k] + X[i,k])/lengthscales[k])**2.
            Kdiag_ref[i] = var * np.exp(-0.5*square_dist) + \
                           var * np.exp(-0.5*square_dist2)

        self.assertTrue(np.allclose(Kxx, Kxx_ref))
        self.assertTrue(np.allclose(Kxx2, Kxx2_ref))
        self.assertTrue(np.allclose(Kdiag, Kdiag_ref))
'''



if __name__ == '__main__':
    unittest.main()
