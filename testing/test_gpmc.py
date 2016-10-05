import numpy as np
import unittest
import tensorflow as tf
import GPflow
import GPinv

class test_gpmc(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = np.linspace(0.,1.,20)
        self.Y = np.sin(3.*self.X) + rng.randn(20)*0.1

    def test_build(self):
        m = GPinv.gpmc.GPMC(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    GPinv.kernels.RBF(1,output_dim=1),
                    GPinv.likelihoods.Gaussian())
        m._compile()

    def test_optimize(self):
        # reference GPR
        m_ref = GPflow.gpr.GPR(
                    self.X.reshape(-1,1),
                    self.Y.reshape(-1,1),
                    kern = GPflow.kernels.RBF(1))
        rslt_ref = m_ref.optimize()
        # tested StVGP
        tf.set_random_seed(1)
        m = GPinv.gpmc.GPMC(self.X.reshape(-1,1), self.Y.reshape(-1,1),
                    kern = GPinv.kernels.RBF(1,output_dim=1),
                    likelihood=GPinv.likelihoods.Gaussian())
        trainer = tf.train.AdamOptimizer(learning_rate=0.002)
        # first optimize by scipy
        m.optimize()
        m.sample(10, verbose=True, epsilon=0.12, Lmax=15)
        Xnew = np.linspace(0.,1.,22)
        mu, var = m.predict_f(Xnew.reshape(-1,1))
        # Just call sample_F and sample_Y
        f_sample = m.sample_F(1)
        y_sample = m.sample_Y(1)

if __name__ == '__main__':
    unittest.main()
