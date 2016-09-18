from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
from GPflow import gpr
from GPinv.multilatent_conditionals import conditional
from GPinv import kernels, densities, transforms
from GPinv.param import Param
from GPinv.multilatent_models import ModelInput, ModelInputSet
from GPinv.multilatent_gpmc import MultilatentGPMC
from GPinv.multilatent_svgp import MultilatentSVGP
from GPinv.nonlinear_model import SVGP
from GPinv.likelihoods import Gaussian, MinibatchGaussian
from GPinv.multilatent_likelihoods import MultilatentLikelihood, MultilatentIndependent
import GPinv

class SingleGaussian(MultilatentLikelihood):
    def __init__(self, num_samples=20):
        MultilatentLikelihood.__init__(self, num_samples)
        self.variance = Param(1., transforms.positive)

    def getCholeskyOf(self, cov):
        var = tf.batch_matrix_diag(tf.batch_matrix_diag_part(cov))
        return tf.sqrt(var)

    def transform(self, F_list):
        return F_list[0]

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)


class DoubleLikelihood(MultilatentIndependent):
    def __init__(self, num_samples=20):
        MultilatentIndependent.__init__(self, num_samples)
        self.variance = Param(1., transforms.positive)

    def transform(self, F_list):
        return F_list[0] + F_list[1]

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)


class test_single(unittest.TestCase):
    """
    In this test, we make sure the multilatent model works propery with a single
    latent function.
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        self.num_params = 20
        self.noise_var = 0.1
        self.X = np.linspace(0, 6., self.num_params).reshape(-1,1)
        self.Y = 2.*np.sin(self.X) + rng.randn(self.num_params).reshape(-1,1) * np.sqrt(self.noise_var)
        # reference GPR
        self.m_ref = gpr.GPR(self.X, self.Y, kern=kernels.RBF(1))
        self.m_ref.optimize()
        tf.set_random_seed(1)

    def test_gpmc(self):
        # tested svgp
        # define the model_input
        model_input1 = ModelInput(self.X, kernels.RBF(1))
        model_input_set = ModelInputSet([model_input1], jitter=1.0e-4)
        # define the model
        m_gpmc = MultilatentGPMC(model_input_set, self.Y,
                            likelihood=SingleGaussian(num_samples=100))
        samples = m_gpmc.sample(num_samples=500, Lmax=20, epsilon=0.05, verbose=False)
        noise = []
        for s in samples[300:]:
            m_gpmc.set_state(s)
            noise.append(m_gpmc.likelihood.variance.value)
        noise_avg = np.mean(noise)
        print(noise_avg)
        self.assertTrue(np.allclose(noise_avg,self.noise_var,rtol=0.2))

    def test_svgp(self):
        # in this test, MultilatentSVGP should work as a simple svgp.
        model_input1 = ModelInput(self.X, kernels.RBF(1))
        model_input_set = ModelInputSet([model_input1],
                                        q_shape='fullrank')
        # define the model
        m_mlvgp = MultilatentSVGP(model_input_set, self.Y,
                            likelihood=SingleGaussian(num_samples=40))
        m_mlvgp.Z.fixed = True
        m_mlvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=2000)
        self.assertTrue(np.allclose(
            self.m_ref._objective(self.m_ref.get_free_state())[0],
            m_mlvgp._objective(m_mlvgp.get_free_state())[0],
            atol = 2.))
        self.assertTrue(np.allclose(
            self.m_ref.kern.lengthscales.value, m_mlvgp.kern.kern_list[0].lengthscales.value,
            rtol = 0.2))
        self.assertTrue(np.allclose(
            self.m_ref.kern.variance.value, m_mlvgp.kern.kern_list[0].variance.value,
            rtol = 0.2))
        self.assertTrue(np.allclose(
            self.m_ref.likelihood.variance.value, m_mlvgp.likelihood.variance.value,
            rtol = 0.2))

    def test_svgp_inducing(self):
        # in this test, MultilatentSVGP should work as a simple svgp.
        model_input1 = ModelInput(self.X, kernels.RBF(1),
                            Z=np.linspace(0.0,6.,10).reshape(-1,1))
        model_input_set = ModelInputSet([model_input1],
                                        q_shape='fullrank')
        # define the model
        m_mlvgp = MultilatentSVGP(model_input_set, self.Y,
                            likelihood=SingleGaussian(num_samples=40))
        # just make sure it works without error
        m_mlvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=10)
        #print(self.m_ref._objective(self.m_ref.get_free_state())[0],
        #        m_mlvgp._objective(m_mlvgp.get_free_state())[0])
        '''
        print(self.m_ref.kern)
        print(m_mlvgp.kern)
        print(self.m_ref.likelihood)
        print(m_mlvgp.likelihood)
        self.assertTrue(np.allclose(
            self.m_ref._objective(self.m_ref.get_free_state())[0],
            m_mlvgp._objective(m_mlvgp.get_free_state())[0],
            atol = 2.))
        self.assertTrue(np.allclose(
            self.m_ref.kern.lengthscales.value, m_mlvgp.kern.kern_list[0].lengthscales.value,
            rtol = 0.2))
        self.assertTrue(np.allclose(
            self.m_ref.kern.variance.value, m_mlvgp.kern.kern_list[0].variance.value,
            rtol = 0.2))
        self.assertTrue(np.allclose(
            self.m_ref.likelihood.variance.value, m_mlvgp.likelihood.variance.value,
            rtol = 0.2))
        '''

class test_double(unittest.TestCase):
    """
    In this test, we consider the following case
    y = f1 + f2 + n
    where y is an observation,
    f1 is a latent function with kernel K1
    f2 is a latent function with kernel K2
    n is an independent noise

    This case can be rewritten as
    y = f + n
    where f follows N(0, K1+K2).

    The latter one can be solved with vanilla GPR.
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        self.noise_var = 0.03
        self.X = np.linspace(0, 9., 60).reshape(-1,1)
        self.Y = 0.5*np.cos(2.*self.X) + 1.5*np.cos(0.5*self.X) + 1. \
                    + rng.randn(60,1)*np.sqrt(self.noise_var)
        # reference GPR
        self.m_ref = gpr.GPR(self.X, self.Y,
                kern=kernels.RBF(1)+kernels.RBF(1))
        self.m_ref.kern.rbf_1.lengthscales=0.8
        self.m_ref.kern.rbf_2.lengthscales=2.
        self.m_ref.optimize()
        tf.set_random_seed(1)

    def test_gpmc(self):
        # tested svgp
        # define the model_input
        rbf = kernels.RBF(1)
        rbf.lengthscales = 0.8
        model_input1 = ModelInput(self.X, rbf)
        model_input2 = ModelInput(self.X, kernels.RBF(1))
        model_input_set = ModelInputSet([model_input1, model_input2], jitter=1.0e-4)
        # define the model
        m_gpmc = MultilatentGPMC(model_input_set, self.Y,
                            likelihood=DoubleLikelihood(num_samples=100))
        samples = m_gpmc.sample(num_samples=1000, Lmax=20, epsilon=0.01, verbose=False)
        noise = []
        for s in samples[500:]:
            m_gpmc.set_state(s)
            noise.append(m_gpmc.likelihood.variance.value)
        noise_avg = np.mean(noise)
        print(noise_avg)
        self.assertTrue(np.allclose(noise_avg,self.noise_var, rtol=0.2))

    def test_svgp(self):
        minibatch_size=30
        rbf = kernels.RBF(1)
        rbf.lengthscales = 0.4
        rbf2 = kernels.RBF(1)
        rbf2.lengthscales = 6.0
        model_input1 = ModelInput(self.X, rbf, X_minibatch_size=minibatch_size,
                                Z=np.linspace(0.5,8.5,30).reshape(-1,1))
        model_input2 = ModelInput(self.X, rbf2, X_minibatch_size=minibatch_size,
                                Z=np.linspace(0.5,8.5,30).reshape(-1,1))
        model_input_set = ModelInputSet([model_input1, model_input2],
                                        q_shape='fullrank')
        # define the model
        m_stvgp = MultilatentSVGP(model_input_set, self.Y,
                            likelihood=DoubleLikelihood(num_samples=50),
                            minibatch_size=minibatch_size)
        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.01), maxiter=2000)
        obj = m_stvgp._objective(m_stvgp.get_free_state())[0]
        print(self.m_ref._objective(self.m_ref.get_free_state())[0])
        print(obj)
        #self.assertTrue(np.allclose(
        #                self.m_ref._objective(self.m_ref.get_free_state())[0],
        #                obj, atol=2.))
        print(self.m_ref.kern)
        print(m_stvgp.kern)
        #self.assertTrue(np.allclose(self.m_ref.kern.rbf_1.lengthscales.value,
        #                     m_stvgp.kern.kern_list[0].lengthscales.value,
        #                     rtol=0.3))
        #self.assertTrue(np.allclose(self.m_ref.kern.rbf_2.lengthscales.value,
        #                     m_stvgp.kern.kern_list[1].lengthscales.value,
        #                     rtol=0.3))
        #self.assertTrue(np.allclose(self.m_ref.kern.rbf_1.variance.value,
        #                     m_stvgp.kern.kern_list[0].variance.value,
        #                     rtol=0.3))
        #self.assertTrue(np.allclose(self.m_ref.kern.rbf_2.variance.value,
        #                     m_stvgp.kern.kern_list[1].variance.value,
        #                     rtol=0.3))
        print(self.m_ref.likelihood)
        print(m_stvgp.likelihood)
        self.assertTrue(np.allclose(self.m_ref.likelihood.variance.value,
                                  m_stvgp.likelihood.variance.value,
                             rtol=0.3))
        #m_svgp = MultilatentSVGP(model_input_set, self.Y,
        #                    likelihood=DoubleLikelihood(num_samples=100))

if __name__ == '__main__':
    unittest.main()
