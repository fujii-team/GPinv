from __future__ import print_function
import GPinv
from GPinv.kernels import RBF,RBF_csym,RBF_casym,Stack
import GPflow
import numpy as np
import unittest
import tensorflow as tf

def ref_kronecker(A, B):
    # straight implementation of kronecker's product
    m = A.shape[0]
    n = A.shape[1]
    p = B.shape[0]
    q = B.shape[1]
    AB = np.zeros((m*p,n*q))
    for i in range(m):
        for j in range(n):
            AB[i*p:(i+1)*p, j*q:(j+1)*q] = A[i,j]*B
    return AB

class test_kronecker(unittest.TestCase):
    def test(self):
        rng = np.random.RandomState(0)
        X = rng.randn(3,2)
        Y = rng.randn(5,4)
        # evaluate XY
        XY = tf.Session().run(GPinv.kronecker.kronecker_product(X, Y))
        # reference kronecker
        self.assertTrue(np.allclose(XY, ref_kronecker(X, Y)))

class test_RBF2D(unittest.TestCase):
    def test(self):
        rng = np.random.RandomState(0)
        X1 = rng.randn(3)
        X2 = rng.randn(5)
        X = np.ndarray((X1.shape[0]*X2.shape[0],2))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                X[i*X2.shape[0]+j,0] = X1[i]
                X[i*X2.shape[0]+j,1] = X2[j]
        X = X.reshape(-1,2)
        # prepare model
        m = GPinv.param.Parameterized()
        m.kronecker = GPinv.kronecker.RBF2D(input_dim=1, output_dim=1,
                        dim1=X1.reshape(-1,1), dim2=X2.reshape(-1,1))
        m.rbf = GPinv.kernels.RBF(input_dim=2, output_dim=1)
        m.make_tf_array(m.get_free_state())
        with m.tf_mode():
            kern = tf.Session().run(m.rbf.K(X), m.get_feed_dict())
            chol = tf.Session().run(m.kronecker.Cholesky(X), m.get_feed_dict())

        for i in range(chol.shape[2]):
            kern_from_chol = np.dot(chol[:,:,i], np.transpose(chol[:,:,i]))
            self.assertTrue(np.allclose(kern[:,:,i], kern_from_chol, atol=1.0e-4))

if __name__ == '__main__':
    unittest.main()
