from __future__ import print_function
import GPinv
from GPinv.kernels import RBF,RBF_csym,RBF_casym,Stack
import GPflow
import numpy as np
import unittest
import tensorflow as tf

class RefRBF(object):
    """
    Reference class for block_diagonal_kernel with rbf kernel.
    """
    def __init__(self, kern):
        self.kern = kern
        self.R = self.kern.variance.value.shape[0]

    def K(self, X, X2=None):
        """
        X and X2 is np.array
        """
        if X2 is None:
            X2 = X

        core = np.zeros((X.shape[0], X2.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X2.shape[0]):
                length = self.kern.lengthscales.value
                x_dif = (X[i] - X2[j])/length
                sq_dist = np.sum(x_dif*x_dif)
                core[i, j] = np.exp(-0.5 * sq_dist)
        K_mat = np.zeros((X.shape[0],X2.shape[0],self.R))
        for k in range(self.R):
            K_mat[:,:,k] = core[:,:] * self.kern.variance.value[k]
        return K_mat

    def Kdiag(self, X):
        K_diag = np.zeros((X.shape[0], self.kern.variance.value.shape[0]))
        for i in range(X.shape[0]):
            var = self.kern.variance.value
            K_diag[i,:] = var
        return K_diag

class RefRBF_csym(RefRBF):
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X2= np.abs(X2)
        X = np.abs(X)
        return RefRBF.K(self,X,X2)+RefRBF.K(self,X,-X2)

    def Kdiag(self, X):
        X = np.abs(X)
        K_diag = np.zeros((X.shape[0], self.kern.variance.value.shape[0]))
        for i in range(X.shape[0]):
            var = self.kern.variance.value
            x_dif = 2*X[i]/self.kern.lengthscales.value
            sq_dist = np.sum(x_dif*x_dif)
            K_diag[i,:] = var * (1.+np.exp(-0.5*sq_dist))
        return K_diag


class RefRBF_casym(RefRBF):
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X2= np.abs(X2)
        X = np.abs(X)
        return RefRBF.K(self,X,X2)-RefRBF.K(self,X,-X2)

    def Kdiag(self, X):
        X = np.abs(X)
        K_diag = np.zeros((X.shape[0], self.kern.variance.value.shape[0]))
        for i in range(X.shape[0]):
            var = self.kern.variance.value
            x_dif = 2*X[i]/self.kern.lengthscales.value
            sq_dist = np.sum(x_dif*x_dif)
            K_diag[i,:] = var * (1.-np.exp(-0.5*sq_dist))
        return K_diag


class test_rbf(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        var = np.exp(rng.randn(3))
        self.m = GPflow.param.Parameterized()
        self.m.kern = RBF(2, 3, var, ARD=True)
        self.m.kern_csym = RBF_csym(2, 3, var, ARD=True)
        self.m.kern_casym = RBF_casym(2, 3, var, ARD=True)
        self.X  = rng.randn(10,2)
        self.X2 = rng.randn(11,2)
        # prepare Parameterized
        tf_array = self.m.get_free_state()
        self.m.make_tf_array(tf_array)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def test_Kxx(self):
        with self.m.tf_mode():
            Kxx  = self.sess.run(self.m.kern.K(self.X))
        ref_rbf = RefRBF(self.m.kern)
        Kxx_ref = ref_rbf.K(self.X)
        self.assertTrue(np.allclose(Kxx, Kxx_ref))

    def test_Kxx2(self):
        with self.m.tf_mode():
            Kxx2 = self.sess.run(self.m.kern.K(self.X,self.X2))
        ref_rbf = RefRBF(self.m.kern)
        Kxx2_ref = ref_rbf.K(self.X,self.X2)
        self.assertTrue(np.allclose(Kxx2, Kxx2_ref))

    def test_Kdiag(self):
        with self.m.tf_mode():
            Kdiag = self.sess.run(self.m.kern.Kdiag(self.X))
        ref_rbf = RefRBF(self.m.kern)
        Kdiag_ref = ref_rbf.Kdiag(self.X)
        self.assertTrue(np.allclose(Kdiag, Kdiag_ref))

    def test_Cholesky(self):
        with self.m.tf_mode():
            K = self.sess.run(self.m.kern.K(self.X))
            chol = self.sess.run(self.m.kern.Cholesky(self.X))
        for i in range(chol.shape[2]):
            self.assertTrue(np.allclose(K[:,:,i],
                        np.dot(chol[:,:,i], np.transpose(chol[:,:,i]))))
    # ------ test for RBF_csym -----
    def test_Kxx_csym(self):
        with self.m.tf_mode():
            Kxx  = self.sess.run(self.m.kern_csym.K(self.X))
        ref_rbf = RefRBF_csym(self.m.kern)
        Kxx_ref = ref_rbf.K(self.X)
        self.assertTrue(np.allclose(Kxx, Kxx_ref))

    def test_Kxx2_csym(self):
        with self.m.tf_mode():
            Kxx2 = self.sess.run(self.m.kern_csym.K(self.X,self.X2))
        ref_rbf = RefRBF_csym(self.m.kern)
        Kxx2_ref = ref_rbf.K(self.X,self.X2)
        self.assertTrue(np.allclose(Kxx2, Kxx2_ref))

    def test_Kdiag_csym(self):
        with self.m.tf_mode():
            Kdiag = self.sess.run(self.m.kern_csym.Kdiag(self.X))
        ref_rbf = RefRBF_csym(self.m.kern)
        Kdiag_ref = ref_rbf.Kdiag(self.X)
        self.assertTrue(np.allclose(Kdiag, Kdiag_ref))

    def test_Cholesky_csym(self):
        with self.m.tf_mode():
            K = self.sess.run(self.m.kern_csym.K(self.X))
            chol = self.sess.run(self.m.kern_csym.Cholesky(self.X))
        for i in range(chol.shape[2]):
            self.assertTrue(np.allclose(K[:,:,i],
                        np.dot(chol[:,:,i], np.transpose(chol[:,:,i]))))
    # ------ test for RBF_casym -----
    def test_Kxx_casym(self):
        with self.m.tf_mode():
            Kxx  = self.sess.run(self.m.kern_casym.K(self.X))
        ref_rbf = RefRBF_casym(self.m.kern)
        Kxx_ref = ref_rbf.K(self.X)
        self.assertTrue(np.allclose(Kxx, Kxx_ref))

    def test_Kxx2_casym(self):
        with self.m.tf_mode():
            Kxx2 = self.sess.run(self.m.kern_casym.K(self.X,self.X2))
        ref_rbf = RefRBF_casym(self.m.kern)
        Kxx2_ref = ref_rbf.K(self.X,self.X2)
        self.assertTrue(np.allclose(Kxx2, Kxx2_ref))

    def test_Kdiag_casym(self):
        with self.m.tf_mode():
            Kdiag = self.sess.run(self.m.kern_casym.Kdiag(self.X))
        ref_rbf = RefRBF_casym(self.m.kern)
        Kdiag_ref = ref_rbf.Kdiag(self.X)
        self.assertTrue(np.allclose(Kdiag, Kdiag_ref))

    def test_Cholesky_casym(self):
        with self.m.tf_mode():
            K = self.sess.run(self.m.kern_casym.K(self.X))
            chol = self.sess.run(self.m.kern_casym.Cholesky(self.X))
        for i in range(chol.shape[2]):
            self.assertTrue(np.allclose(K[:,:,i],
                        np.dot(chol[:,:,i], np.transpose(chol[:,:,i])),
                        atol=1.0e-4))

class test_stack(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        var = np.exp(rng.randn(3))
        self.m = GPflow.param.Parameterized()
        self.kern1 = RBF(2, 3, var, ARD=True)
        self.kern2 = RBF(2, 3, var, ARD=True)
        self.kern3 = RBF_csym(2, 3, var, ARD=True)
        self.m.kern =Stack([self.kern1,self.kern2,self.kern3,])
        self.X  = rng.randn(10,2)
        self.X2 = rng.randn(11,2)
        # prepare Parameterized
        tf_array = self.m.get_free_state()
        self.m.make_tf_array(tf_array)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def test_Kxx(self):
        with self.m.tf_mode():
            Kxx  = self.sess.run(self.m.kern.K(self.X))
        ref1 = RefRBF(self.kern1)
        ref2 = RefRBF(self.kern2)
        ref3 = RefRBF_csym(self.kern3)
        self.assertTrue(np.allclose(Kxx[:,:,0:3], ref1.K(self.X)))
        self.assertTrue(np.allclose(Kxx[:,:,3:6], ref2.K(self.X)))
        self.assertTrue(np.allclose(Kxx[:,:,6:9], ref3.K(self.X)))

    def test_Kxx2(self):
        with self.m.tf_mode():
            Kxx2 = self.sess.run(self.m.kern.K(self.X, self.X2))
        ref1 = RefRBF(self.kern1)
        ref2 = RefRBF(self.kern2)
        ref3 = RefRBF_csym(self.kern3)
        self.assertTrue(np.allclose(Kxx2[:,:,0:3], ref1.K(self.X, self.X2)))
        self.assertTrue(np.allclose(Kxx2[:,:,3:6], ref2.K(self.X, self.X2)))
        self.assertTrue(np.allclose(Kxx2[:,:,6:9], ref3.K(self.X, self.X2)))

    def test_Kdiag(self):
        with self.m.tf_mode():
            Kdiag = self.sess.run(self.m.kern.Kdiag(self.X))
        ref1 = RefRBF(self.kern1)
        ref2 = RefRBF(self.kern2)
        ref3 = RefRBF_csym(self.kern3)
        self.assertTrue(np.allclose(Kdiag[:,0:3], ref1.Kdiag(self.X)))
        self.assertTrue(np.allclose(Kdiag[:,3:6], ref2.Kdiag(self.X)))
        self.assertTrue(np.allclose(Kdiag[:,6:9], ref3.Kdiag(self.X)))

    def test_Cholesky(self):
        with self.m.tf_mode():
            Kxx  = self.sess.run(self.m.kern.K(self.X))
            chol = self.sess.run(self.m.kern.Cholesky(self.X))
        for i in range(chol.shape[2]):
            self.assertTrue(np.allclose(Kxx[:,:,i],
                        np.dot(chol[:,:,i], np.transpose(chol[:,:,i]))))



if __name__ == '__main__':
    unittest.main()
