from __future__ import print_function
import os
import os.path
import tensorflow as tf
import numpy as np
import unittest
import GPinv
import GPflow

class Quadratic(GPinv.model.StVmodel):
    def __init__(self):
        GPinv.model.StVmodel.__init__(self, GPinv.kernels.RBF(1,1),
                    GPinv.mean_functions.Zero(1),GPinv.likelihoods.Gaussian())
        self.rng = np.random.RandomState(0)
        self.x = GPinv.param.LocalParam(self.rng.randn(2,1))
        self.y = GPflow.param.Param(self.rng.randn(2,1))
        self.m = GPflow.param.Parameterized()
        self.m.z = GPflow.param.Param(self.rng.randn(2,1))
        self.m.x= GPinv.param.LocalParam(self.rng.randn(2,1))

    def build_likelihood(self):
        return -tf.reduce_sum(tf.square(self.x - np.ones((2,1))) + \
                              tf.square(self.y - np.ones((2,1))*0.1)+ \
                              tf.square(self.m.z - np.ones((2,1))*0.01) +\
                              tf.square(self.m.x - np.ones((2,1))*0.001))

class test_timeline(unittest.TestCase):
    def setUp(self):
        self.m = Quadratic()
    def test(self):
        trainer = tf.train.AdadeltaOptimizer(learning_rate=0.01)
        filename = 'timeline.json'
        if os.path.exists(filename):
            os.remove(filename)
        # normal optimize does not write timeline
        self.m.optimize(trainer, maxiter=10)
        self.assertFalse(os.path.exists(filename))
        # edit custom config to trace timeline
        custom_config = GPflow.settings.get_settings()
        custom_config.profiling.dump_timeline = True
        with GPflow.settings.temp_settings(custom_config):
            self.m.optimize(trainer, maxiter=10)
            self.assertTrue(os.path.exists(filename))
        os.remove(filename)
        # make sure the custom filename works.
        filename2 = '.timeline.json'
        custom_config.profiling.timeline_file = filename2
        with GPflow.settings.temp_settings(custom_config):
            self.m.optimize(trainer, maxiter=10)
            self.assertTrue(os.path.exists(filename2))
        # clean up these files
        os.remove(filename2)


if __name__ == '__main__':
    unittest.main()
