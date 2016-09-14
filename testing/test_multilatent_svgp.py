from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
from GPflow import gpr
from GPinv import kernels, densities, transforms
from GPinv.param import Param
from GPinv.multilatent_svgp import MultilatentSVGP
from GPinv.multilatent_param import ModelInput
from GPinv.likelihoods import MultilatentLikelihood
import GPinv

class DoubleLikelihood(MultilatentLikelihood):
    def __init__(self, num_samples=20):
        MultilatentLikelihood.__init__(self, num_samples)
        self.variance = Param(1.0, transforms.positive)

    def transform(self, F_list):
        return F_list[0] + F_list[1]

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)


class test(unittest.TestCase):
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
        self.X = np.linspace(0, 6., 40).reshape(-1,1)
        self.Y = 1.5*np.cos(self.X) + 3. + rng.randn(40,1)*0.3

    def test_svgp(self):
        # reference GPR
        m_ref = gpr.GPR(self.X, self.Y,
                kern=kernels.RBF(1)+kernels.Constant(1))
        m_ref.optimize()
        print(m_ref)
        # tested svgp
        # define the model_input
        Z1 = np.linspace(0, 6., 10).reshape(-1,1)
        Z2 = np.linspace(0, 6., 2).reshape(-1,1)
        model_input1 = ModelInput(self.X, kernels.RBF(1), Z1.copy())
        model_input2 = ModelInput(self.X, kernels.Constant(1), Z2.copy())
        # define the model
        m_svgp = MultilatentSVGP([model_input1, model_input2],
                                    self.Y,
                                    likelihood=DoubleLikelihood(num_samples=100),
                                    num_latent=1,
                                    q_shape='fullrank')
        m_svgp.kern.kernel_list[0][0].lengthscale = 0.1
        # optimize
        m_svgp.optimize(tf.train.AdamOptimizer(learning_rate=0.02), maxiter=5000)

        print(m_ref.kern)
        print(m_svgp.kern)

        print(m_ref.likelihood)
        print(m_svgp.likelihood)

        print(m_ref.predict_f(self.X)[0].flatten())
        print(m_svgp.predict_f(m_svgp.X.concat())[0].flatten())

if __name__ == '__main__':
    unittest.main()
