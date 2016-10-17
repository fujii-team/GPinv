from __future__ import print_function
import GPinv
from GPinv.mean_functions import Zero, Constant, Stack
import GPflow
import numpy as np
import unittest
import tensorflow as tf

class test_mean(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        var = np.exp(rng.randn(3))
        self.m = GPflow.param.Parameterized()
        self.m.mean1 = Zero(3)
        self.m.mean2 = Constant(3)
        self.m.mean3 = Constant(3, c=np.ones(3)*0.1)
        self.X  = rng.randn(10,2)
        # prepare Parameterized
        tf_array = self.m.get_free_state()
        self.m.make_tf_array(tf_array)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def test_zero(self):
        with self.m.tf_mode():
            mean  = self.sess.run(self.m.mean1(self.X))
        self.assertTrue(np.allclose(mean, np.zeros((3,10))))

    def test_const(self):
        with self.m.tf_mode():
            mean2 = self.sess.run(self.m.mean2(self.X))
            mean3 = self.sess.run(self.m.mean3(self.X))
        self.assertTrue(np.allclose(mean2, np.ones((3,10))))
        self.assertTrue(np.allclose(mean3, np.ones((3,10))*0.1))

class test_stack(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        var = np.exp(rng.randn(3))
        self.m = GPflow.param.Parameterized()
        self.mean1 = Zero(3)
        self.mean2 = Constant(3)
        self.mean3 = Constant(3, c=np.ones(3)*0.1)
        self.m.mean = Stack([self.mean1,self.mean2,self.mean3])
        self.X  = rng.randn(10,2)
        # prepare Parameterized
        tf_array = self.m.get_free_state()
        self.m.make_tf_array(tf_array)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def test(self):
        with self.m.tf_mode():
            mean = self.sess.run(self.m.mean(self.X))
        self.assertTrue(np.allclose(mean[0:3,:], np.zeros((3,10))))
        self.assertTrue(np.allclose(mean[3:6,:], np.ones((3,10))))
        self.assertTrue(np.allclose(mean[6:9,:], np.ones((3,10))*0.1))


if __name__ == '__main__':
    unittest.main()
