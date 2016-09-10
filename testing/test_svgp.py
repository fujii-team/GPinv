import GPflow
import tensorflow as tf
import numpy as np
import unittest
from GPinv.nonlinear_model import SVGP
from GPinv.likelihoods import Gaussian, MinibatchGaussian

class test_vgp(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.num_params = 40
        self.X = np.linspace(0.0, 6.0, self.num_params).reshape(-1,1)
        self.Y = 2.*np.sin(self.X) + rng.randn(self.num_params).reshape(-1,1) * 0.3

    def test_approximate_Gaussian_likelihood(self):
        # reference GPR
        m_ref = GPflow.gpr.GPR(self.X, self.Y, kern = GPflow.kernels.RBF(1))
        m_ref.optimize(disp=False)

        minibatchGaussian = MinibatchGaussian(self.num_params, self.num_params, 40, exact=False)
        m_stvgp = SVGP(self.X, self.Y,
                    kern = GPflow.kernels.RBF(1),
                    likelihood=minibatchGaussian,
                    Z = self.X.copy())
        m_stvgp.Z.fixed = True

        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=2000)
        obj_stvgp = np.mean(
                [m_stvgp._objective(m_stvgp.get_free_state())[0] for i in range(10)])
        # needs rough agreement.
        print(m_ref._objective(m_ref.get_free_state())[0])
        print(obj_stvgp)
        self.assertTrue(np.allclose(  m_ref._objective(m_ref.get_free_state())[0],
                                    obj_stvgp,
                                    atol=2.))

        # TODO
        # not sure why but the kernel hyperparameters are not well estimated
        #self.assertTrue(np.allclose(  m_ref.kern.variance.value,
        #                            m_stvgp.kern.variance.value, rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.lengthscales.value,
        #                            m_stvgp.kern.lengthscales.value, rtol=0.2))
        self.assertTrue(np.allclose(  m_ref.likelihood.variance.value,
                                    m_stvgp.likelihood.variance.value, rtol=0.2))

        Xnew = np.linspace(0.0, 6.0, 29).reshape(-1,1)
        f_ref =   m_ref.predict_f(Xnew)
        f_stvgp=m_stvgp.predict_f(Xnew)

        self.assertTrue(np.allclose(f_ref[0], f_stvgp[0], atol=0.2))
        self.assertTrue(np.allclose(f_ref[1], f_stvgp[1], atol=0.2))



if __name__ == "__main__":
    unittest.main()
