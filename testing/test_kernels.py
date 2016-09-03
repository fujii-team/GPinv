from __future__ import print_function
from GPinv.kernels import Zero, RBF_csym
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


if __name__ == '__main__':
    unittest.main()
