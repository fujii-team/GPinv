from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
from GPflow import gpr
from GPinv import kernels, densities, transforms
from GPinv.param import Param
from GPinv.multilatent_models import ModelInput, ModelInputSet
from GPinv.multilatent_gpmc import MultilatentGPMC
#from GPinv.multilatent_svgp import MultilatentSVGP
from GPinv.likelihoods import MultilatentLikelihood
import GPinv

class SingleGaussian(MultilatentLikelihood):
    def __init__(self, num_samples=20):
        MultilatentLikelihood.__init__(self, num_samples)
        self.variance = Param(1., transforms.positive)

    def transform(self, F_list):
        return F_list[0]

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)

class DoubleLikelihood(MultilatentLikelihood):
    def __init__(self, num_samples=20):
        MultilatentLikelihood.__init__(self, num_samples)
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
        self.X = np.linspace(0, 6., 60).reshape(-1,1)
        self.Y = 1.5*np.cos(1.*self.X) + rng.randn(60,1)*0.3
        # reference GPR
        self.m_ref = gpr.GPR(self.X, self.Y, kern=kernels.RBF(1))
        self.m_ref.optimize()

    def test_gpmc(self):
        # tested svgp
        # define the model_input
        model_input1 = ModelInput(self.X, kernels.RBF(1))
        model_input_set = ModelInputSet([model_input1], jitter=1.0e-4)
        # define the model
        m_gpmc = MultilatentGPMC(model_input_set, self.Y,
                            likelihood=SingleGaussian(num_samples=100))
        samples = m_gpmc.sample(num_samples=1000, Lmax=20, epsilon=0.01, verbose=False)
        noise = []
        for s in samples[500:]:
            m_gpmc.set_state(s)
            noise.append(m_gpmc.likelihood.variance.value)
        noise_avg = np.mean(noise)
        print(noise_avg)
        self.assertTrue(np.allclose(noise_avg,
                                    self.m_ref.likelihood.variance.value,
                                            rtol=0.2))

    def _test_svgp(self):
        minibatch_size=30
        model_input1 = ModelInput(self.X, kernels.RBF(1), X_minibatch_size=minibatch_size,
                                Z=np.linspace(0.5,5.5,20).reshape(-1,1))
        model_input_set = ModelInputSet([model_input1],
                                        q_shape='fullrank')
        # define the model
        m_stvgp = MultilatentSVGP(model_input_set, self.Y,
                            likelihood=SingleGaussian(num_samples=100),
                            minibatch_size=minibatch_size)
        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=3000)
        print(self.m_ref._objective(self.m_ref.get_free_state())[0])
        print(m_stvgp._objective(m_stvgp.get_free_state())[0])
        print(m_stvgp.kern)
        print(m_stvgp.likelihood)
        #m_svgp = MultilatentSVGP(model_input_set, self.Y,
        #                    likelihood=DoubleLikelihood(num_samples=100))



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
        self.X = np.linspace(0, 6., 60).reshape(-1,1)
        self.Y = 1.5*np.cos(4.*self.X) + 0.5*np.cos(0.5*self.X) + 1. \
                    + rng.randn(60,1)*0.3
        # reference GPR
        self.m_ref = gpr.GPR(self.X, self.Y,
                kern=kernels.RBF(1)+kernels.RBF(1))
        self.m_ref.kern.rbf_1.lengthscales=0.1
        self.m_ref.optimize()

    def test_gpmc(self):
        # tested svgp
        # define the model_input
        rbf = kernels.RBF(1)
        rbf.lengthscales = 0.3
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
        self.assertTrue(np.allclose(noise_avg,
                                    self.m_ref.likelihood.variance.value,
                                            rtol=0.2))

    def _test_1svgp(self):
        minibatch_size=30
        rbf = kernels.RBF(1)
        rbf.lengthscales = 0.3
        rbf2 = kernels.RBF(1)
        rbf2.lengthscales = 3.0
        model_input1 = ModelInput(self.X, rbf, X_minibatch_size=minibatch_size,
                                Z=np.linspace(0.5,5.5,20).reshape(-1,1))
        model_input2 = ModelInput(self.X, rbf2, X_minibatch_size=minibatch_size,
                                Z=np.linspace(0.5,5.5,20).reshape(-1,1))
        model_input_set = ModelInputSet([model_input1, model_input2],
                                        q_shape='fullrank')
        # define the model
        m_stvgp = MultilatentSVGP(model_input_set, self.Y,
                            likelihood=DoubleLikelihood(num_samples=100),
                            minibatch_size=minibatch_size)
        m_stvgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=3000)
        print(self.m_ref._objective(self.m_ref.get_free_state())[0])
        print(m_stvgp._objective(m_stvgp.get_free_state())[0])
        print(m_stvgp.kern)
        print(m_stvgp.likelihood)
        #m_svgp = MultilatentSVGP(model_input_set, self.Y,
        #                    likelihood=DoubleLikelihood(num_samples=100))

if __name__ == '__main__':
    unittest.main()
