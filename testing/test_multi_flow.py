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
from GPinv.nonlinear_model import SVGP
from GPinv.multilatent_likelihoods import MultilatentLikelihood, MultilatentIndependent
import GPinv


class DoubleLikelihood(MultilatentIndependent):
    def __init__(self, num_samples=20):
        MultilatentIndependent.__init__(self, num_samples)
        self.variance = Param(1., transforms.positive)

    def transform(self, F_list):
        return F_list[0] + F_list[1]

    def log_p(self, F, Y):
        return densities.gaussian(F, Y, self.variance)

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
        self.rng = np.random.RandomState(0)
        self.noise_var = 0.03
        self.X = np.linspace(0, 9., 20).reshape(-1,1)
        self.X2 = np.linspace(3., 12., 20).reshape(-1,1)
        self.Y = 0.5*np.cos(2.*self.X) + 1.5*np.cos(0.5*self.X) + 1. \
                    + self.rng.randn(20,1)*np.sqrt(self.noise_var)
        # reference GPR
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
        #samples = m_gpmc.sample(num_samples=100, epsilon=0.1, verbose=False)
        m_gpmc.V = self.rng.randn(40,1)
        m_gpmc._compile()
        # prediction
        f_pred , f_var = m_gpmc.predict_f([self.X, self.X])
        f_pred2, f_var2= m_gpmc.predict_f([self.X, self.X2])
        # Check the result differs for different argument.
        #print(f_pred[0])
        #print(f_pred2[0])
        self.assertFalse(np.allclose(f_pred[1], f_pred2[1]))

        g_mu, gmu_minus, gmu_plus = m_gpmc.predict_g()


if __name__ == '__main__':
    unittest.main()
