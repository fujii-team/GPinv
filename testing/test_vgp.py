import GPflow
import tensorflow as tf
import numpy as np
import unittest
from GPinv.vgp import VGP
from GPinv.likelihoods import Gaussian

class test_vgp(unittest.TestCase):
    def test_single(self):
        """
        Test the case where all the likelihood is i.i.d, and kernel is
        block-diagonal, resulting in a usual independent GP.
        """
        # offdiag should be zero for the independent case, but some are tracked
        # here to make sure this model works.
        rng = np.random.RandomState(0)
        num_params = 40
        X = np.linspace(0.0, 3.0, num_params).reshape(-1,1)
        Y = 1.*np.cos(X) + rng.randn(num_params).reshape(-1,1) * 0.3

        # reference GPR
        m_ref = GPflow.gpr.GPR(X, Y, kern = GPflow.kernels.RBF(1))
        m_ref.optimize(disp=False)
        # reference VGP
        m_vgp = GPflow.vgp.VGP(X, Y, kern = GPflow.kernels.RBF(1),
                likelihood=GPflow.likelihoods.Gaussian())
        m_vgp.optimize(disp=False)

        # stochastic vgp with mean-field approximation (true for Gaussian likelihood)
        m_stvgp = VGP(X, Y, kern = GPflow.kernels.RBF(1),
                likelihood=Gaussian(40),
                mode='mean_field')
        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.01), maxiter=500)

        # stochastic vgp with mean-field approximation (true for Gaussian likelihood)
        m_stvgp2 = VGP(X, Y, kern = GPflow.kernels.RBF(1),
                likelihood=Gaussian(40),
                mode='semi_diag',
                semidiag_list=[{'head_index':[20,0], 'length':5}])
        m_stvgp2.optimize(tf.train.AdamOptimizer(learning_rate=0.01), maxiter=500)
        '''
        print(m_ref._objective(m_ref.get_free_state())[0])
        print(m_vgp._objective(m_vgp.get_free_state())[0])
        print(m_stvgp._objective(m_stvgp.get_free_state())[0])
        print(m_stvgp2._objective(m_stvgp2.get_free_state())[0])
        print(m_ref.kern)
        print(m_vgp.kern)
        print(m_stvgp.kern)
        print(m_stvgp2.kern)
        print(m_ref.likelihood)
        print(m_vgp.likelihood)
        print(m_stvgp.likelihood)
        print(m_stvgp2.likelihood)
        #print(m_stvgp2.q_lambda.matrices[0])
        print(m_stvgp2.q_lambda.matrices[1])
        '''
        # needs rough agreement.
        self.assertTrue(np.allclose(  m_ref._objective(m_ref.get_free_state())[0],
                                    m_stvgp._objective(m_stvgp.get_free_state())[0],
                                    rtol=0.2))
        self.assertTrue(np.allclose(  m_ref._objective(m_ref.get_free_state())[0],
                                    m_stvgp2._objective(m_stvgp2.get_free_state())[0],
                                    rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.variance.value,
        #                            m_stvgp.kern.variance.value, rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.variance.value,
        #                            m_stvgp2.kern.variance.value, rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.lengthscales.value,
        #                            m_stvgp.kern.lengthscales.value, rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.lengthscales.value,
        #                            m_stvgp2.kern.lengthscales.value, rtol=0.2))
        self.assertTrue(np.allclose(  m_ref.likelihood.variance.value,
                                    m_stvgp.likelihood.variance.value, rtol=0.2))
        self.assertTrue(np.allclose(  m_ref.likelihood.variance.value,
                                    m_stvgp2.likelihood.variance.value, rtol=0.2))

        Xnew = np.linspace(0.0, 3.0, 29).reshape(-1,1)
        f_ref =   m_ref.predict_f(Xnew)
        f_vgp =   m_vgp.predict_f(Xnew)
        f_stvgp=m_stvgp.predict_f(Xnew)
        f_stvgp2=m_stvgp2.predict_f(Xnew)

        self.assertTrue(np.allclose(f_ref[0], f_vgp[0], atol=0.1))
        self.assertTrue(np.allclose(f_ref[1], f_vgp[1], atol=0.1))

        self.assertTrue(np.allclose(f_ref[0], f_stvgp[0], atol=0.1))
        self.assertTrue(np.allclose(f_ref[1], f_stvgp[1], atol=0.1))

        self.assertTrue(np.allclose(f_ref[0], f_stvgp2[0], atol=0.1))
        self.assertTrue(np.allclose(f_ref[1], f_stvgp2[1], atol=0.1))


    '''
    def test_offdiag(self):
        """
        Test the case where all the likelihood is i.i.d, and kernel is
        block-diagonal, resulting in a usual independent GP.
        """
        # offdiag should be zero for the independent case, but some are tracked
        # here to make sure this model works.
        rng = np.random.RandomState(0)
        num_params = 20
        X = np.linspace(0.0,2.0, num_params).reshape(-1,1)
        Y = np.cos(X) + rng.randn(num_params).reshape(-1,1) * 0.1
        """
        # fullrank vgp
        m_full = VGP(X, Y, kern = GPflow.kernels.RBF(1),
                likelihood=Gaussian(num_params, 1000),
                mode='full_rank')
        m_full.optimize(display=True)
        print(m_full)
        """
        # specified_offdiag vgp
        m_spec = VGP(X, Y, kern = GPflow.kernels.RBF(1),
                likelihood=Gaussian(num_params, 1000),
                mode='specified',
                offdiag_indices=np.array([[1, 0]]))
        m_spec.optimize(display=False)

        print(m_spec._objective(m_spec.get_free_state()))
        print(m_spec)
    '''

if __name__ == "__main__":
    unittest.main()
