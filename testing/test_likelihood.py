import tensorflow as tf
import numpy as np
import GPflow
from GPinv.likelihoods import Gaussian
import unittest


class GaussianLikelihood_ref(object):
    """
    Reference for the test
    """
    def __init__(self, variance=1., num_samples=20):
        self.variance = variance
        self.num_samples = num_samples
        self.rng = np.random.RandomState(0)

    def log_p(self, y, f):
        return -0.5 * np.log(2 * np.pi) \
               - 0.5 * np.log(self.variance)\
               - 0.5 * np.square(f-y)/self.variance

    def stochastic_expectations(self, f, cov, y):
        """
        :param 1d-np.array f: mean of the latent function. Shape [N]
        :param 2d-np.array L: Cholesky factor for the covariance [N,N]
        :return 1d-np.array: stochastic expectations for \int log(y|f)p(f)df
        """
        L = tf.cholesky(cov)
        expectations = np.zeros(f.shape)
        for i in range(self.num_samples):
            f_sample = f+np.dot(L, self.rng.randn(L.shape[1]))
            expectations += self.log_p(f_sample, y) / self.num_samples
        return expectations


class test_Gaussian(unittest.TestCase):
    def test(self):
        tf.set_random_seed(0)
        """
        Stochastic expectation should approach to the exact value
        """
        rng = np.random.RandomState(0)
        num_params = 10

        # make L matrix
        variance_f = 2.
        L = np.zeros((num_params, num_params))
        for i in range(num_params):
            for j in range(i):
                #L[i,j] = 0.3#rng.randn(1)
                pass
            L[i,i] = np.sqrt(variance_f)
        cov = np.dot(L, L.transpose())
        cov_diag = np.diag(np.diagonal(cov))

        variance = 1.
        f = rng.randn(num_params)
        y = np.zeros(num_params)

        m = GPflow.model.Model()
        m.gaussian = Gaussian(num_samples=100,exact=False)
        m.gaussian.variance = variance
        m.gaussian2 = Gaussian(num_samples=300,exact=False)
        m.gaussian2.variance = variance
        m.gaussian3 = Gaussian(num_samples=1000,exact=False)
        m.gaussian3.variance = variance
        m.gaussian_exact = Gaussian(exact=True)
        m.gaussian_exact.variance = variance
        m.F = GPflow.param.DataHolder(f.reshape(-1,1))
        m.Y = GPflow.param.DataHolder(y.reshape(-1,1))
        m.cov = GPflow.param.DataHolder(cov.reshape(num_params,num_params,1))
        m.cov_diag = GPflow.param.DataHolder(cov_diag.reshape(num_params,num_params,1))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.set_random_seed(1)

        tf_array = m.get_free_state()
        m.make_tf_array(tf_array)
        with m.tf_mode():
            expectation = sess.run(
                    m.gaussian.stochastic_expectations(m.F, m.cov, m.Y),
                        feed_dict = m.get_feed_dict())
            expectation2 = sess.run(
                    m.gaussian2.stochastic_expectations(m.F, m.cov, m.Y),
                        feed_dict = m.get_feed_dict())
            expectation3 = sess.run(
                    m.gaussian3.stochastic_expectations(m.F, m.cov, m.Y),
                        feed_dict = m.get_feed_dict())
            expectation_diag = sess.run(
                    m.gaussian3.stochastic_expectations(m.F, m.cov_diag, m.Y),
                        feed_dict = m.get_feed_dict())
            # exact solution, where Fvar is diagonal part of m.L * m.L^T
            expectation_exact = sess.run(
                    m.gaussian_exact.stochastic_expectations(m.F, m.cov, m.Y),
                        feed_dict = m.get_feed_dict())
        # print(expectation)
        # print(expectation2)
        # print(expectation3)
        # print(expectation_exact)
        res = np.mean(expectation - expectation_exact)
        res2= np.mean(expectation2 - expectation_exact)
        res3= np.mean(expectation3 - expectation_exact)
        res_diag= np.mean(expectation_diag - expectation_exact)
        # assert the residiual decreases by increasing sample number.
        self.assertTrue(np.abs(res)> np.abs(res3))
        # assert the approximation is close to the exact values.
        self.assertTrue(np.allclose(expectation3, expectation_exact,
            rtol=0.1))

if __name__ == "__main__":
    unittest.main()
