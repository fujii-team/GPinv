import tensorflow as tf
import GPflow
from GPinv.param import MinibatchData
import numpy as np
import unittest

class test_MinibatchData(unittest.TestCase):
    def test(self):
        rng = np.random.RandomState(0)
        N = 50
        n = 3
        minibatch_size = 5
        array = rng.randn(N,n)
        m = GPflow.Parameterized()
        d = MinibatchData(array, minibatch_size, 0)
        #TODO


if __name__ == "__main__":
    unittest.main()
