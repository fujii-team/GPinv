from __future__ import print_function
import GPinv
from GPinv.kernels import Zero, RBF, RBF_csym, BlockDiagonal
from GPinv.param import ConcatDataHolder, ConcatParamList
from GPinv.multilatent_models import ModelInput, ModelInputSet
import GPflow
import numpy as np
import unittest
import tensorflow as tf

class test_zero(unittest.TestCase):
    def test(self):
        rng = np.random.RandomState(0)

        X = rng.randn(10,2)
        X2= rng.randn(12,2)

        m = GPflow.model.Model()
        m.kern = Zero(2)
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

        self.assertTrue(np.allclose(Kxx, np.zeros((X.shape[0], X.shape[0]))))
        self.assertTrue(np.allclose(Kxx2, np.zeros((X.shape[0], X2.shape[0]))))
        self.assertTrue(np.allclose(Kdiag, np.zeros(X.shape[0])))

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

class ref_rbf(object):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def __init__(self, kern):
        self.kern = kern

    def K(self, X, X2=None):
        """
        X and X2 is np.array
        """
        if X2 is None:
            X2 = X

        K_mat = np.zeros((X.shape[0], X2.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X2.shape[0]):
                var = self.kern.variance.value
                length = self.kern.lengthscales.value
                x_dif = (X[i] - X2[j])/length
                sq_dist = np.sum(x_dif*x_dif)
                K_mat[i, j] = var * np.exp(-0.5 * sq_dist)
        return K_mat

    def Kdiag(self, X):
        K_diag = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            var = self.kern.variance.value
            K_diag[i] = var
        return K_diag


class test_block_diagonal(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X1 = self.rng.randn(10, 2)
        self.Z1 = self.rng.randn( 3, 2)
        self.model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2), self.Z1)

        self.X2 = self.rng.randn(11, 2)
        self.Z2 = self.rng.randn( 4, 2)
        self.model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2), self.Z2)

    def test(self):
        model_input1 = ModelInput(self.X1, RBF(2), self.Z1)
        model_input2 = ModelInput(self.X2, RBF(2), self.Z2)
        model_input_set = ModelInputSet([model_input1, model_input2])

        m = GPflow.model.Model()
        jitter = 1.0e-3
        m.kern = model_input_set.getKernel(jitter=1.0e-3)
        m.X = model_input_set.getConcat_X()
        m.Z = model_input_set.getConcat_Z()

        tf_array = m.get_free_state()
        m.make_tf_array(tf_array)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        with m.tf_mode():
            Kxx  = sess.run(m.kern.K(m.Z), feed_dict = m.get_feed_dict())
            Kxx2 = sess.run(m.kern.K(m.Z, m.X), feed_dict = m.get_feed_dict())
            Kdiag = sess.run(m.kern.Kdiag(m.Z), feed_dict = m.get_feed_dict())
            cholesky = sess.run(m.kern.Cholesky(m.Z), feed_dict = m.get_feed_dict())
        # reference kernel
        kern_ref = ref_rbf(m.kern.kern_list[0])
        # Kxx
        # diagonal block
        self.assertTrue(np.allclose(kern_ref.K(self.Z1, self.Z1), Kxx[:3,:3]))
        self.assertTrue(np.allclose(kern_ref.K(self.Z2, self.Z2), Kxx[3:,3:]))
        # non-diagonal block
        self.assertTrue(np.allclose(np.zeros((self.Z2.shape[0],self.Z1.shape[0])), Kxx[3:,:3]))
        self.assertTrue(np.allclose(np.zeros((self.Z1.shape[0],self.Z2.shape[0])), Kxx[:3,3:]))
        # Kxx2
        # diagonal block
        self.assertTrue(np.allclose(kern_ref.K(self.Z1, self.X1), Kxx2[:3,:10]))
        self.assertTrue(np.allclose(kern_ref.K(self.Z2, self.X2), Kxx2[3:,10:]))
        # non-diagonal block
        self.assertTrue(np.allclose(np.zeros((self.Z2.shape[0],self.X1.shape[0])), Kxx2[3:,:10]))
        self.assertTrue(np.allclose(np.zeros((self.Z1.shape[0],self.X2.shape[0])), Kxx2[:3,10:]))
        # Kdiag
        self.assertTrue(np.allclose(kern_ref.Kdiag(self.Z2), Kdiag[3:]))
        self.assertTrue(np.allclose(kern_ref.Kdiag(self.Z1), Kdiag[:3]))
        # cholesky
        Kxx_cholesky = np.dot(cholesky, np.transpose(cholesky))
        Kxx_jitter = Kxx + np.eye(self.Z1.shape[0]+self.Z2.shape[0])*jitter
        self.assertTrue(np.allclose(Kxx_cholesky, Kxx_jitter))

if __name__ == '__main__':
    unittest.main()
