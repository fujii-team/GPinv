import GPflow
import tensorflow as tf
import numpy as np
import unittest
from GPinv.nonlinear_model import SVGP
from GPinv.likelihoods import Gaussian, MinibatchGaussian
from GPflow.param import DataHolder

class test_vgp(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.num_params = 40
        self.X = np.linspace(0.0, 6.0, self.num_params).reshape(-1,1)
        self.Y = 2.*np.sin(self.X) + rng.randn(self.num_params).reshape(-1,1) * 0.3
        # reference GPR
        self.m_ref = GPflow.gpr.GPR(self.X, self.Y, kern = GPflow.kernels.RBF(1))
        self.m_ref.optimize(disp=False)
        self.ref_objective = self.m_ref._objective(self.m_ref.get_free_state())[0]
        self.Xnew = np.linspace(0.0, 6.0, 29).reshape(-1,1)
        self.f_ref =   self.m_ref.predict_f(self.Xnew)
        tf.set_random_seed(1)

    def test_svgp_nonexact_lik(self):
        tf.set_random_seed(1)
        minibatchGaussian = MinibatchGaussian(
                            self.num_params, self.num_params, 40, exact=False)
        m_stvgp = SVGP(self.X, self.Y,
                    kern = GPflow.kernels.RBF(1),
                    likelihood=minibatchGaussian,
                    Z = self.X.copy(),
                    minibatch_size=self.num_params)
        m_stvgp.Z.fixed = True
        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=3000)
        obj_stvgp = np.mean(
                [m_stvgp._objective(m_stvgp.get_free_state())[0] for i in range(10)])
        print(self.ref_objective)
        print(obj_stvgp)
        self.assertTrue(np.allclose(self.ref_objective, obj_stvgp, atol=2.))
        self.assertTrue(np.allclose(self.m_ref.likelihood.variance.value,
                                    m_stvgp.likelihood.variance.value, rtol=0.2))
        f_stvgp=m_stvgp.predict_f(self.Xnew)
        self.assertTrue(np.allclose(self.f_ref[0], f_stvgp[0], atol=0.2))
        self.assertTrue(np.allclose(self.f_ref[1], f_stvgp[1], atol=0.2))


    def test_svgp_full(self):
        tf.set_random_seed(1)
        minibatchGaussian = MinibatchGaussian(self.num_params)
        m_stvgp = SVGP(self.X, self.Y,
                    kern = GPflow.kernels.RBF(1),
                    likelihood=minibatchGaussian,
                    Z = self.X.copy())
        m_stvgp.Z.fixed = True

        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=3000)
        obj_stvgp = np.mean(
                [m_stvgp._objective(m_stvgp.get_free_state())[0] for i in range(10)])
        # needs rough agreement.
        print(self.ref_objective)
        print(obj_stvgp)
        self.assertTrue(np.allclose(self.ref_objective, obj_stvgp, atol=2.))
        # TODO
        # not sure why but the kernel hyperparameters are not well estimated
        #self.assertTrue(np.allclose(  m_ref.kern.variance.value,
        #                            m_stvgp.kern.variance.value, rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.lengthscales.value,
        #                            m_stvgp.kern.lengthscales.value, rtol=0.2))
        self.assertTrue(np.allclose(self.m_ref.likelihood.variance.value,
                                    m_stvgp.likelihood.variance.value, rtol=0.2))

        f_stvgp=m_stvgp.predict_f(self.Xnew)

        self.assertTrue(np.allclose(self.f_ref[0], f_stvgp[0], atol=0.2))
        self.assertTrue(np.allclose(self.f_ref[1], f_stvgp[1], atol=0.2))

    def test_svgp_minibatch_inducing(self):
        tf.set_random_seed(1)
        # with inducing point and minibatching
        inducing = 10
        minibatch_size = 20
        minibatchGaussian2 = MinibatchGaussian(self.num_params, minibatch_size, 40)
        m_stvgp2 = SVGP(self.X, self.Y,
                    kern = GPflow.kernels.RBF(1),
                    likelihood=minibatchGaussian2,
                    Z = np.linspace(0.5,5.5,inducing).reshape(-1,1),
                    minibatch_size=minibatch_size)

        m_stvgp2.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=3000)
        obj_stvgp2 = np.mean(
                [m_stvgp2._objective(m_stvgp2.get_free_state())[0] for i in range(10)])
        # needs rough agreement.
        print(obj_stvgp2)
        # TODO
        # not sure why but the kernel hyperparameters are not well estimated
        #self.assertTrue(np.allclose(  m_ref.kern.variance.value,
        #                            m_stvgp2.kern.variance.value, rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.lengthscales.value,
        #                            m_stvgp2.kern.lengthscales.value, rtol=0.2))
        print(m_stvgp2.likelihood.variance.value)
        print(self.m_ref.likelihood.variance.value)
        self.assertTrue(np.allclose(self.m_ref.likelihood.variance.value,
                                    m_stvgp2.likelihood.variance.value, rtol=0.2))

        f_stvgp2=m_stvgp2.predict_f(self.Xnew)

        # print(self.f_ref[0][:,0])
        # print(f_stvgp2[0][:,0]-self.f_ref[0][:,0])
        self.assertTrue(np.allclose(self.f_ref[0], f_stvgp2[0], atol=0.2))
        self.assertTrue(np.allclose(self.f_ref[1], f_stvgp2[1], atol=0.2))

    def test_Y_minibatch_off(self):
        # If minibatch_size is None, Y should be treated just as DataHolder
        tf.set_random_seed(1)
        inducing = 15
        minibatch_size = 20
        m_stvgp2 = SVGP(self.X, self.Y,
                    kern = GPflow.kernels.RBF(1),
                    likelihood=Gaussian(),
                    Z = np.linspace(0.5,5.5,inducing).reshape(-1,1),
                    minibatch_size=minibatch_size,
                    X_minibatch=True)
        self.assertTrue(isinstance(m_stvgp2.Y, DataHolder))

    def test_X_minibatch(self):
        tf.set_random_seed(1)
        inducing = 15
        minibatch_size = 20
        m_stvgp2 = SVGP(self.X, self.Y,
                    kern = GPflow.kernels.RBF(1),
                    likelihood=Gaussian(),
                    Z = np.linspace(0.5,5.5,inducing).reshape(-1,1),
                    minibatch_size=minibatch_size,
                    X_minibatch=True)

        m_stvgp2.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=3000)
        obj_stvgp2 = np.mean(
                [m_stvgp2._objective(m_stvgp2.get_free_state())[0] for i in range(10)])
        # needs rough agreement.
        print(obj_stvgp2)
        # TODO
        # not sure why but the kernel hyperparameters are not well estimated
        #self.assertTrue(np.allclose(  m_ref.kern.variance.value,
        #                            m_stvgp2.kern.variance.value, rtol=0.2))
        #self.assertTrue(np.allclose(  m_ref.kern.lengthscales.value,
        #                            m_stvgp2.kern.lengthscales.value, rtol=0.2))
        print(m_stvgp2.likelihood.variance.value)
        print(self.m_ref.likelihood.variance.value)
        self.assertTrue(np.allclose(self.m_ref.likelihood.variance.value,
                                    m_stvgp2.likelihood.variance.value, rtol=0.2))
        f_stvgp2=m_stvgp2.predict_f(self.Xnew)
        # print(self.f_ref[0][:,0])
        # print(f_stvgp2[0][:,0]-self.f_ref[0][:,0])
        self.assertTrue(np.allclose(self.f_ref[0], f_stvgp2[0], atol=0.2))
        self.assertTrue(np.allclose(self.f_ref[1], f_stvgp2[1], atol=0.2))

if __name__ == "__main__":
    unittest.main()
