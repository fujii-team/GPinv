from __future__ import print_function
from GPinv.kernels import Zero, RBF_csym, BlockDiagonalKernel
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

class ref_block_diagonal_kernel(object):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def __init__(self, rbf_kern_list):
        self.kern_list = rbf_kern_list

    def K(self, X, X2=None):
        """
        X and X2 is np.array
        """
        if X2 is None:
            X2 = X

        K_mat = np.zeros((X.shape[0], X2.shape[0]))
        for i in range(X.shape[0]):
            ind = int(X[i, -1])
            for j in range(X2.shape[0]):
                ind2 = int(X2[j, -1])
                if ind == ind2:
                    var = self.kern_list[ind].variance.value
                    length = self.kern_list[ind].lengthscales.value
                    x_dif = (X[i, :-1] - X2[j, :-1])/length
                    sq_dist = np.sum(x_dif*x_dif)
                    K_mat[i, j] = \
                            var * np.exp(-0.5 * sq_dist)
        return K_mat



class test_block_diagonal_kernel(unittest.TestCase):
    def test(self):
        rng = np.random.RandomState(0)

        X = np.hstack([rng.randn(10,2), rng.randint(0,2,10).reshape(-1,1)])
        X2= np.hstack([rng.randn(12,2), rng.randint(0,2,12).reshape(-1,1)])

        kernels = [GPflow.kernels.RBF(2),GPflow.kernels.RBF(2)]
        kernels[1].variance = 2.0
        kernels[1].lengthscales = 2.0
        kern_ref = ref_block_diagonal_kernel(kernels)

        m = GPflow.model.Model()
        m.kern = BlockDiagonalKernel(kernels)
        m.X = GPflow.param.DataHolder(X)
        m.X2= GPflow.param.DataHolder(X2)

        tf_array = m.get_free_state()
        m.make_tf_array(tf_array)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        with m.tf_mode():
            Kxx  = sess.run(m.kern.K(m.X), feed_dict = m.get_feed_dict())
            Kxx2 = sess.run(m.kern.K(m.X, m.X2), feed_dict = m.get_feed_dict())

        self.assertTrue(np.allclose(Kxx, kern_ref.K(X, X)))
        self.assertTrue(np.allclose(Kxx2, kern_ref.K(X, X2)))

if __name__ == '__main__':
    unittest.main()
