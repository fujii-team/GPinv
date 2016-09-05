from __future__ import print_function
from GPinv.mean_functions import SwitchedMeanFunction
import GPflow
import numpy as np
import unittest
import tensorflow as tf

class TestSwitchedMeanFunction(unittest.TestCase):
    """
    Test for the SwitchedMeanFunction.
    """
    def test(self):
        rng = np.random.RandomState(0)
        X = np.hstack([rng.randn(10,3), 1.0*rng.randint(0,2,10).reshape(-1,1)])
        switched_mean = SwitchedMeanFunction(
                        [GPflow.mean_functions.Constant(np.zeros(1)),
                         GPflow.mean_functions.Constant(np.ones( 1))] )

        sess = tf.Session()
        tf_array = switched_mean.get_free_state()
        switched_mean.make_tf_array(tf_array)
        sess.run(tf.initialize_all_variables())
        with switched_mean.tf_mode():
            result = sess.run(
                    switched_mean(X), feed_dict=switched_mean.get_feed_dict())

        np_list=np.array([0.,1.])
        result_ref = (np_list[X[:,3].astype(np.int)]).reshape(-1,1)
        self.assertTrue(np.allclose(result, result_ref))

if __name__ == '__main__':
    unittest.main()
