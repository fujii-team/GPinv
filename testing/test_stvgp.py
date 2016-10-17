import numpy as np
import unittest
import tensorflow as tf
import GPflow
import GPinv
from GPflow._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

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


    def test_transform_sample(self):
        rng = np.random.RandomState(0)
        N,R,n = 2,1,self.X.shape[0]
        # function to test for 1 option
        def func(q_shape, q_sqrt):
            m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian(),
                    q_shape=q_shape, num_samples=N)
            m._q_sqrt = q_sqrt
            m._compile()
            with m.tf_mode():
                v_samples = tf.random_normal([R,n,N], dtype=float_type)
                mu = tf.expand_dims(m.q_mu, -1)
                u_ref_tf = mu + tf.batch_matmul(m.q_sqrt, v_samples)
                u_tra_tf = m._transform_samples(v_samples)
                # evaluate logdet
                logdet_ref_tf = tf.reduce_sum(
                        tf.log(tf.square(tf.batch_matrix_diag_part(m.q_sqrt))))
                logdet_tra_tf = m._logdet()
                u_ref, u_tra, logdet_ref, logdet_tra = \
                    m._session.run([u_ref_tf, u_tra_tf, logdet_ref_tf, logdet_tra_tf])
            self.assertTrue(np.allclose(u_ref, u_tra))
            self.assertTrue(np.allclose(logdet_ref, logdet_tra))

        # diagonal
        q_sqrt = np.exp(rng.randn(R,n))
        func('diagonal', q_sqrt)
        # fullrank
        q_sqrt = rng.randn(R,n,n)
        func('fullrank', q_sqrt)
        # q_shape=3
        q_sqrt = np.transpose(np.hstack(
                        [np.ones((n,R))*3.0,np.ones((n,R))*2.0, np.ones((n,R))]
                        ).reshape(n,3,R), [2,0,1])
        func(3, q_sqrt)


    def test_q_shape_diag(self):
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian(),
                    q_shape='diagonal')
        m._compile()
        with m.tf_mode():
            sqrt = m._session.run(m.q_sqrt)
        self.assertTrue(np.allclose(
                        sqrt, np.diag(np.ones(20)).reshape(1,20,20)))

    def test_q_shape_multidiag(self):
        X = np.ones((5,1))
        Y = np.ones((5,1))
        m = GPinv.stvgp.StVGP(X.reshape(-1,1), Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian(),
                    q_shape=3)
        m._q_sqrt = np.transpose(np.hstack(
                        [np.ones((5,1))*3.0,np.ones((5,1))*2.0, np.ones((5,1))]
                        ).reshape(5,3,1), [2,0,1])
        sqrt_ref = np.eye(5)
        for i in range(5-1):
            sqrt_ref[i+1,i] = 2.0
        for i in range(5-2):
            sqrt_ref[i+2,i] = 3.0
        m._compile()
        with m.tf_mode():
            sqrt = m._session.run(m.q_sqrt)
        self.assertTrue(np.allclose(sqrt[0,:,:], sqrt_ref))

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
        #print(mu.flatten())
        #print(mu_ref.flatten())
        #print(var.flatten())
        #print(var_ref.flatten())
        self.assertTrue(np.allclose(mu, mu_ref, atol=0.03))
        self.assertTrue(np.allclose(var, var_ref, atol=0.003))

    def test_sample_from(self):
        # tested StVGP
        tf.set_random_seed(1)
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian())
        # get samplesk
        n_sample = 10
        f_samples = m.sample_from('F', n_sample)
        self.assertTrue(np.allclose(f_samples.shape, [1,self.X.shape[0], n_sample]))
        self.assertTrue(hasattr(m, '_sample_from_F_AF_storage'))
        f_samples = m.sample_from('F', n_sample)


    def test_samples(self):
        # tested StVGP
        tf.set_random_seed(1)
        m = GPinv.stvgp.StVGP(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian())
        # get samples
        num_samples = 10
        f_samples = m.sample_from('F',num_samples)
        self.assertTrue(np.allclose(f_samples.shape, [1,self.X.shape[0],num_samples]))
        y_samples = m.sample_from('Y',num_samples)
        self.assertTrue(np.allclose(y_samples.shape, [1,self.X.shape[0],num_samples]))

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
