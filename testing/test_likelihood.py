import tensorflow as tf
import numpy as np
import GPflow
from GPinv.likelihoods import Gaussian
import unittest


class GaussianLikelihood_ref(object):
    """
    Reference for the test
    """
    def __init__(self, variance=1., num_stocastic_points=20):
        self.variance = variance
        self.num_stocastic_points = num_stocastic_points
        self.rng = np.random.RandomState(0)

    def logp(self, y, f):
        return -0.5 * np.log(2 * np.pi) \
               - 0.5 * np.log(self.variance)\
               - 0.5 * np.square(f-y)/self.variance

    def stochastic_expectations(self, f, L, y):
        """
        :param 1d-np.array f: mean of the latent function. Shape [N]
        :param 2d-np.array L: Cholesky factor for the covariance [N,N]
        :return 1d-np.array: stochastic expectations for \int log(y|f)p(f)df
        """
        expectations = np.zeros(f.shape)
        for i in range(self.num_stocastic_points):
            f_sample = f+np.dot(L, self.rng.randn(L.shape[1]))
            expectations += self.logp(f_sample, y) / self.num_stocastic_points
        return expectations


class test_Gaussian(unittest.TestCase):
    def test(self):
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
        L_diag = np.diag(np.sqrt(np.diag(np.dot(L, L.transpose()))))

        variance = 1.
        ref = GaussianLikelihood_ref(variance=variance, num_stocastic_points=100)
        ref2 = GaussianLikelihood_ref(variance=variance, num_stocastic_points=300)
        ref3 = GaussianLikelihood_ref(variance=variance, num_stocastic_points=1000)
        f = np.zeros(num_params)
        y = np.zeros(num_params)

        m = GPflow.model.Model()
        m.gaussian = Gaussian(num_stocastic_points=100,exact=False)
        m.gaussian.variance = variance
        m.gaussian2 = Gaussian(num_stocastic_points=300,exact=False)
        m.gaussian2.variance = variance
        m.gaussian3 = Gaussian(num_stocastic_points=1000,exact=False)
        m.gaussian3.variance = variance
        m.gaussian_exact = Gaussian(exact=True)
        m.gaussian_exact.variance = variance
        m.F = GPflow.param.DataHolder(f.reshape(-1,1))
        m.Y = GPflow.param.DataHolder(y.reshape(-1,1))
        m.L = GPflow.param.DataHolder(L.reshape(num_params,num_params,1))
        m.L_diag = GPflow.param.DataHolder(L_diag.reshape(num_params,num_params,1))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        tf.set_random_seed(1)

        tf_array = m.get_free_state()
        m.make_tf_array(tf_array)
        with m.tf_mode():
            expectation_ref = ref.stochastic_expectations(f, L, y)
            expectation_ref2 = ref2.stochastic_expectations(f, L, y)
            expectation_ref3 = ref3.stochastic_expectations(f, L, y)
            expectation = sess.run(
                    m.gaussian.stochastic_expectations(m.F, m.L, m.Y),
                        feed_dict = m.get_feed_dict())
            expectation2 = sess.run(
                    m.gaussian2.stochastic_expectations(m.F, m.L, m.Y),
                        feed_dict = m.get_feed_dict())
            expectation3 = sess.run(
                    m.gaussian3.stochastic_expectations(m.F, m.L, m.Y),
                        feed_dict = m.get_feed_dict())
            expectation_diag = sess.run(
                    m.gaussian3.stochastic_expectations(m.F, m.L_diag, m.Y),
                        feed_dict = m.get_feed_dict())
            # exact solution, where Fvar is diagonal part of m.L * m.L^T
            expectation_exact = sess.run(
                    m.gaussian_exact.stochastic_expectations(m.F, m.L, m.Y),
                        feed_dict = m.get_feed_dict())
        print(expectation_ref)
        print(expectation)
        print(expectation2)
        print(expectation3)
        res_ref = np.mean(expectation_ref - expectation_exact)
        res_ref2 = np.mean(expectation_ref2 - expectation_exact)
        res_ref3 = np.mean(expectation_ref3 - expectation_exact)
        res = np.mean(expectation - expectation_exact)
        res2= np.mean(expectation2 - expectation_exact)
        res3= np.mean(expectation3 - expectation_exact)
        res_diag= np.mean(expectation_diag - expectation_exact)
        print(res_ref, res_ref2, res_ref3, res, res2, res3, res_diag)
        # assert the residiual decreases by increasing sample number.
        self.assertTrue(np.abs(res) > np.abs(res2))
        self.assertTrue(np.abs(res2)> np.abs(res3))
        # assert the approximation is close to the exact values.
        self.assertTrue(np.allclose(np.sum(expectation), np.sum(expectation_exact),
            rtol=1.0e-2))

if __name__ == "__main__":
    unittest.main()
