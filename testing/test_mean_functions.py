from __future__ import print_function
import GPinv
import GPflow
from GPinv.mean_functions import Constant, Zero, SwitchedMeanFunction
from GPinv.param import ConcatDataHolder, ConcatParamList
from GPinv.multilatent_models import ModelInput, ModelInputSet
import numpy as np
import unittest
import tensorflow as tf

class TestSwitchedMeanFunction(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X1 = self.rng.randn(10, 2)
        self.Z1 = self.rng.randn( 3, 2)
        self.mean1 = Constant(np.ones(2)*2.)

        self.X2 = self.rng.randn(11, 2)
        self.Z2 = self.rng.randn( 4, 2)
        self.mean2 = Constant(np.ones(2)*3.)

    def test(self):
        model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2), self.Z1,
                                                    mean_function=self.mean1)
        model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2), self.Z2,
                                                    mean_function=self.mean2)
        model_input_set = ModelInputSet([model_input1, model_input2])

        m = GPflow.model.Model()
        m.mean = model_input_set.getMeanFunction()
        m.mean1 = self.mean1
        m.mean2 = self.mean2
        m.X = model_input_set.getConcat_X()

        tf_array = m.get_free_state()
        m.make_tf_array(tf_array)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        with m.tf_mode():
            means = sess.run(m.mean(m.X), feed_dict=m.get_feed_dict())
            mean1 = sess.run(m.mean1(self.X1), feed_dict=m.get_feed_dict())
            mean2 = sess.run(m.mean2(self.X2), feed_dict=m.get_feed_dict())
        # reference kernel
        self.assertTrue(np.allclose(means[:10,:], mean1))
        self.assertTrue(np.allclose(means[10:,:], mean2))


if __name__ == '__main__':
    unittest.main()
