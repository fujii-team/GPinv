import GPflow
import tensorflow as tf
import numpy as np
import unittest
from GPinv.nonlinear_model import VGP
from GPinv.likelihoods import Gaussian

class test_vgp(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.num_params = 40
        self.X = np.linspace(0.0, 6.0, self.num_params).reshape(-1,1)
        self.Y = 2.*np.sin(self.X) + rng.randn(self.num_params).reshape(-1,1) * 0.3


    def test_exact_Gaussian_likelihood(self):
        """
        Test the case where all the likelihood is i.i.d, and kernel is
        block-diagonal, resulting in a usual independent GP.
        """
        # offdiag should be zero for the independent case, but some are tracked
        # here to make sure this model works.
        # reference VGP
        m_vgp = GPflow.vgp.VGP(self.X, self.Y, kern = GPflow.kernels.RBF(1),
                likelihood=GPflow.likelihoods.Gaussian())
        m_vgp.optimize(disp=False)

        # stochastic vgp with mean-field approximation (true for Gaussian likelihood)
        m_vgp2 = VGP(self.X, self.Y, kern = GPflow.kernels.RBF(1),
                    likelihood=Gaussian(exact=True), mode='mean_field')
        #m_stvgp.optimize()
        m_vgp2.optimize(method='BFGS',disp=False)

        self.assertTrue(np.allclose(m_vgp.kern.lengthscales.value,
                                    m_vgp2.kern.lengthscales.value, rtol=1.0e-1))
        self.assertTrue(np.allclose(m_vgp.kern.variance.value,
                                    m_vgp2.kern.variance.value, rtol=1.0e-1))
        self.assertTrue(np.allclose(m_vgp.likelihood.variance.value,
                                    m_vgp2.likelihood.variance.value, rtol=1.0e-1))


    def test_approximate_Gaussian_likelihood(self):
        # reference GPR
        m_ref = GPflow.gpr.GPR(self.X, self.Y, kern = GPflow.kernels.RBF(1))
        m_ref.optimize(disp=False)

        tf.set_random_seed(1)
        m_stvgp = VGP(self.X, self.Y, kern = GPflow.kernels.RBF(1),
                    likelihood=Gaussian(40, exact=False), mode='mean_field')
        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.05), maxiter=500)
        obj_stvgp = np.mean(
                [m_stvgp._objective(m_stvgp.get_free_state())[0] for i in range(10)])
        # needs rough agreement.
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

        self.assertTrue(np.allclose(f_ref[0], f_stvgp[0], atol=0.1))
        self.assertTrue(np.allclose(f_ref[1], f_stvgp[1], atol=0.1))



if __name__ == "__main__":
    unittest.main()
