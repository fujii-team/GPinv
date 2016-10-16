import numpy as np
import unittest
import tensorflow as tf
import GPflow
import GPinv

class test_stvgp(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = np.linspace(0.,1.,20)
        self.Y = np.sin(3.*self.X) + rng.randn(20)*0.1

    def test_build(self):
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    GPinv.kernels.RBF(1,output_dim=1),
                    GPinv.likelihoods.Gaussian())
        m._compile()

    def test_q_shape_diag(self):
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian(),
                    q_shape='diagonal')
        m._compile()
        with m.tf_mode():
            sqrt = m._session.run(m.q_sqrt)
        self.assertTrue(np.allclose(
                        sqrt, np.diag(np.ones(20)).reshape(20,20,1)))

    def test_q_shape_multidiag(self):
        X = np.ones((5,1))
        Y = np.ones((5,1))
        m = GPinv.stvgp.StVGP(X.reshape(-1,1), Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian(),
                    q_shape=3)
        m._q_sqrt = np.hstack(
                        [np.ones((5,1)),np.ones((5,1))*2.0, np.ones((5,1))*3.0]
                        ).reshape(5,3,1)
        sqrt_ref = np.eye(5)
        for i in range(5-1):
            sqrt_ref[i+1,i] = 2.0
        for i in range(5-2):
            sqrt_ref[i+2,i] = 3.0
        m._compile()
        with m.tf_mode():
            sqrt = m._session.run(m.q_sqrt)
        self.assertTrue(np.allclose(sqrt[:,:,0], sqrt_ref))

    def test_optimize(self):
        # reference GPR
        m_ref = GPflow.gpr.GPR(
                    self.X.reshape(-1,1),
                    self.Y.reshape(-1,1),
                    kern = GPflow.kernels.RBF(1))
        rslt_ref = m_ref.optimize()
        # tested StVGP
        tf.set_random_seed(1)
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian())
        trainer = tf.train.AdamOptimizer(learning_rate=0.002)
        # first optimize by scipy
        m.optimize()
        # Stochastic optimization by tf.train
        rslt = m.optimize(trainer, maxiter=3000)
        print(rslt['fun'], rslt_ref['fun'])
        self.assertTrue(np.allclose(rslt['fun'], rslt_ref['fun'], atol=0.5))
        print(m.kern)
        print(m_ref.kern)
        print(m.likelihood)
        print(m_ref.likelihood)
        self.assertTrue(np.allclose(
            m.kern.lengthscales.value, m_ref.kern.lengthscales.value, rtol=0.2))
        self.assertTrue(np.allclose(
            m.kern.variance.value, m_ref.kern.variance.value, rtol=0.4))
        self.assertTrue(np.allclose(
            m.likelihood.variance.value, m_ref.likelihood.variance.value, rtol=0.2))
        # test prediction
        Xnew = np.linspace(0.,1.,22)
        mu, var = m.predict_f(Xnew.reshape(-1,1))
        mu_ref, var_ref = m_ref.predict_f(Xnew.reshape(-1,1))
        print(mu.flatten())
        print(mu_ref.flatten())
        print(var.flatten())
        print(var_ref.flatten())
        self.assertTrue(np.allclose(mu, mu_ref, atol=0.03))
        self.assertTrue(np.allclose(var, var_ref, atol=0.003))

    def test_sample_from(self):
        # tested StVGP
        tf.set_random_seed(1)
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian())
        # get samples
        n_sample = 10
        f_samples = m.sample_from_('sample_F', n_sample)
        self.assertTrue(np.allclose(f_samples.shape, [n_sample,self.X.shape[0], 1]))



    def test_samples(self):
        # tested StVGP
        tf.set_random_seed(1)
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian())
        # get samples
        num_samples = 10
        f_samples = m.sample_F(num_samples)
        self.assertTrue(np.allclose(f_samples.shape, [num_samples,self.X.shape[0], 1]))
        y_samples = m.sample_Y(num_samples)
        self.assertTrue(np.allclose(y_samples.shape, [num_samples,self.X.shape[0], 1]))

    def test_KL_analytic(self):
        """
        Test option for stvgp.KL_analytic
        """
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    GPinv.kernels.RBF(1,output_dim=1),
                    GPinv.likelihoods.Gaussian(),
                    KL_analytic = True)
        trainer = tf.train.AdamOptimizer(learning_rate=0.002)
        # Stochastic optimization by tf.train
        rslt = m.optimize(trainer, maxiter=3000)

    def test_qdiag(self):
        """
        Test option for stvgp.KL_analytic
        """
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    GPinv.kernels.RBF(1,output_dim=1),
                    GPinv.likelihoods.Gaussian(),
                    q_shape = 'diagonal')
        trainer = tf.train.AdamOptimizer(learning_rate=0.002)
        # Stochastic optimization by tf.train
        rslt = m.optimize(trainer, maxiter=3000)


if __name__ == '__main__':
    unittest.main()
