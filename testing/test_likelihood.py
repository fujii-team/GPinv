import tensorflow as tf
import numpy as np
import GPflow
from GPinv.likelihoods import Gaussian
import unittest

class test_Gaussian(unittest.TestCase):
    def test(self):
        """
        Stochastic expectation should approach to the exact value
        if Fvar < likelihood.variance
        """
        rng = np.random.RandomState(0)
        num_params = 10

        # make L matrix
        variance = 10.0
        L = np.zeros((num_params, num_params,1))
        for i in range(num_params):
            for j in range(i):
                L[i,j,0] = rng.randn(1)
            L[i,i,0] = np.sqrt(variance)

        m = GPflow.model.Model()
        m.gaussian = Gaussian(num_stocastic_points=1000)
        m.gaussian.variance = variance
        m.gaussian_exact = GPflow.likelihoods.Gaussian()
        m.gaussian_exact.variance = variance
        m.F = GPflow.param.DataHolder(rng.randn(num_params).reshape(-1,1))
        m.Y = GPflow.param.DataHolder(rng.randn(num_params).reshape(-1,1))
        m.L = GPflow.param.DataHolder(L)
        m.Fvar = GPflow.param.DataHolder(np.diag(
            np.dot(L[:,:,0], L[:,:,0].transpose())).reshape(-1,1))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.set_random_seed(1)

        tf_array = m.get_free_state()
        m.make_tf_array(tf_array)
        with m.tf_mode():
            expectation = sess.run(
                    m.gaussian.stochastic_expectations(
                        m.F, m.L, m.Y),
                        feed_dict = m.get_feed_dict())
            # exact solution, where Fvar is diagonal part of m.L * m.L^T
            expectation_exact = sess.run(
                    m.gaussian_exact.variational_expectations(
                        m.F, m.Fvar, m.Y),
                        feed_dict = m.get_feed_dict())

        self.assertTrue(np.allclose(np.sum(expectation), np.sum(expectation_exact),
            rtol=1.0e-2))

if __name__ == "__main__":
    unittest.main()
