from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import GPflow
from GPinv.param import DataHolder
from GPinv.multilatent_likelihoods import MultilatentLikelihood, MultilatentIndependent

class test_mlIndependent(unittest.TestCase):
    def setUp(self):
        """
        Consider [2,4x5, 4x5] tensor, which simulates the covariance of
        5 channel data with 4 latent functions.
        """
        rng = np.random.RandomState(1)
        tmp = rng.randn(4*5,4*5)
        self.mat_concat = np.zeros((2,4*5,4*5))
        self.mat_concat[0,:,:] = np.eye(4*5)#+np.dot(tmp, np.transpose(tmp))
        self.mat_concat[1,:,:] = np.eye(4*5)#+np.dot(tmp, np.transpose(tmp))
        self.div_mat = np.zeros((2, 5, 4, 4))
        self.block_mat = np.zeros((2,4*5,4*5))
        self.slice_begin = []
        self.slice_size = []
        for i in range(4):
            self.slice_begin.append(i*5)
            self.slice_size.append(5)
            for j in range(4):
                for I in range(5):
                    self.div_mat[0,I,i,j] = self.mat_concat[0,5*i+I,5*j+I]
                    self.div_mat[1,I,i,j] = self.mat_concat[1,5*i+I,5*j+I]
                    self.block_mat[0,5*i+I,5*j+I] = self.mat_concat[0,5*i+I,5*j+I]
                    self.block_mat[1,5*i+I,5*j+I] = self.mat_concat[1,5*i+I,5*j+I]
        tmp = np.linalg.cholesky(self.mat_concat)
        self.block_chol = np.linalg.cholesky(self.block_mat)

    def test_reshapeCov(self):
        m = GPflow.model.Model()
        m.lik = MultilatentIndependent(jitter=0.0)
        m.lik.make_slice_indices(self.slice_begin, self.slice_size)
        m.cov = DataHolder(self.mat_concat)

        m.make_tf_array(m.get_free_state())
        with m.tf_mode():
            diag_cov = m._session.run(m.lik.getBlockDiagCov(m.cov),
                                            feed_dict = m.get_feed_dict())
            diag_chol = m._session.run(m.lik.getCholeskyOf(m.cov),
                                            feed_dict = m.get_feed_dict())
        self.assertTrue(np.allclose(self.div_mat, diag_cov))
        self.assertTrue(np.allclose(self.block_chol, diag_chol))

if __name__ == '__main__':
    unittest.main()
