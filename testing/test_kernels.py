from __future__ import print_function
import GPinv
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
    def test(self):
        pass


if __name__ == '__main__':
    unittest.main()
