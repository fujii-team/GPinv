import tensorflow as tf
import GPflow
import GPinv
from GPinv import transforms
from GPinv.param import ConcatDataHolder, ConcatParamList, SqrtParamList
from GPinv.multilatent_models import ModelInput, ModelInputSet
import numpy as np
import unittest

class test_multilatent_param(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X1 = self.rng.randn(10, 2)
        self.Z1 = self.rng.randn( 3, 2)

        self.X2 = self.rng.randn(10, 2)
        self.Z2 = self.rng.randn( 3, 2)

        # manually generate X concat to make sure it aligns correctly.
        self.X_concat = np.ndarray((20,2))
        for i in range(len(self.X1)):
            self.X_concat[i,:] = self.X1[i,:]
        for i in range(len(self.X2)):
            self.X_concat[i+10,:] = self.X2[i,:]
        # manually generate X concat to make sure it aligns correctly.
        self.Z_concat = np.ndarray((6,2))
        for i in range(len(self.Z1)):
            self.Z_concat[i,:] = self.Z1[i,:]
        for i in range(len(self.Z2)):
            self.Z_concat[i+3,:] = self.Z2[i,:]


    def test_ConcatDataHolder(self):
        model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2))
        model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2))
        input_set = ModelInputSet([model_input1, model_input2])

        m = GPflow.model.Model()
        m.concatData = input_set.getConcat_X()
        with m.tf_mode():
            X = m._session.run(tf.identity(m.concatData), feed_dict = m.get_feed_dict())

        self.assertTrue(np.allclose(self.X_concat, X))
        # test getitem
        self.assertTrue(np.allclose(self.X_concat, m.concatData.concat()))
        # test setitem
        # make sure assigning.
        X2 = self.rng.randn(10,2)
        m.concatData[1] = X2
        self.assertTrue(np.allclose(X2, m.concatData[1]))

    def test_ConcatParamList(self):
        m = GPflow.model.Model()
        m.concatParam = ConcatParamList([self.Z1, self.Z2])
        m.make_tf_array(m.get_free_state())
        with m.tf_mode():
            Z = m._session.run(tf.identity(m.concatParam.concat()), feed_dict = m.get_feed_dict())
        self.assertTrue(np.allclose(self.Z_concat, Z))
        self.assertTrue(np.allclose(self.Z_concat, m.concatParam.concat()))
        self.assertTrue(np.allclose(m.concatParam.shape, self.Z_concat.shape))

    def test_ConcatParamList_InputSet(self):
        model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2), Z=self.Z1)
        model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2), Z=self.Z2)
        input_set = ModelInputSet([model_input1, model_input2])
        m = GPflow.model.Model()
        m.concatParam = input_set.getConcat_Z()
        m.make_tf_array(m.get_free_state())
        with m.tf_mode():
            Z = m._session.run(tf.identity(m.concatParam.concat()), feed_dict = m.get_feed_dict())
        self.assertTrue(np.allclose(self.Z_concat, Z))
        self.assertTrue(np.allclose(self.Z_concat, m.concatParam.concat()))

    def test_ConcatParamList_InputSetDefault(self):
        model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2))
        model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2))
        input_set = ModelInputSet([model_input1, model_input2])
        m = GPflow.model.Model()
        m.concatParam = input_set.getConcat_Z()
        m.make_tf_array(m.get_free_state())
        with m.tf_mode():
            Z = m._session.run(tf.identity(m.concatParam.concat()), feed_dict = m.get_feed_dict())
        self.assertTrue(np.allclose(self.X_concat, Z))
        self.assertTrue(np.allclose(self.X_concat, m.concatParam.concat()))

    def test_SqrtParamList_full(self):
        model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2))
        model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2))
        # with default values
        input_set = ModelInputSet([model_input1, model_input2],
                                q_shape='fullrank')
        # with specified values
        q_list = [self.rng.randn(10,10,1), self.rng.randn(10,10,1), self.rng.randn(10,10,1)]
        input_set2 = ModelInputSet([model_input1, model_input2],
                                q_shape='fullrank', q_sqrt_list=q_list)
        m = GPflow.model.Model()
        m.sqrtParam = input_set.getConcat_q_sqrt()
        m.sqrtParam2 = input_set2.getConcat_q_sqrt()
        m.make_tf_array(m.get_free_state())
        with m.tf_mode():
            qsqrt  = m._session.run(tf.identity(m.sqrtParam.concat()), feed_dict = m.get_feed_dict())
            qsqrt2 = m._session.run(tf.identity(m.sqrtParam2.concat()), feed_dict = m.get_feed_dict())

        q_sqrt_ref = np.zeros((20,20,1))
        for i in range(20):
            q_sqrt_ref[i,i,0] = 1.
        q_sqrt_ref2 = np.zeros((20,20,1))
        q_sqrt_ref2[:10, :10] = q_list[0]
        q_sqrt_ref2[10:, :10] = q_list[1]
        q_sqrt_ref2[10:, 10:] = q_list[2]
        self.assertTrue(np.allclose(q_sqrt_ref, qsqrt))
        self.assertTrue(np.allclose(q_sqrt_ref2, qsqrt2))
        self.assertTrue(np.allclose(q_sqrt_ref, m.sqrtParam.concat()))
        self.assertTrue(np.allclose(q_sqrt_ref2, m.sqrtParam2.concat()))

    def test_SqrtParamList_diag(self):
        model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2))
        model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2))
        # with default values
        q_list = [np.exp(self.rng.randn(10,1)), np.exp(self.rng.randn(10,1))]
        input_set = ModelInputSet([model_input1, model_input2],
                                q_shape='diagonal')
        input_set2 = ModelInputSet([model_input1, model_input2],
                                q_shape='diagonal', q_sqrt_list=q_list)
        m = GPflow.model.Model()
        m.sqrtParam = input_set.getConcat_q_sqrt()
        m.sqrtParam2 = input_set2.getConcat_q_sqrt()
        m.make_tf_array(m.get_free_state())
        with m.tf_mode():
            qsqrt  = m._session.run(tf.identity(m.sqrtParam.concat()), feed_dict = m.get_feed_dict())
            qsqrt2 = m._session.run(tf.identity(m.sqrtParam2.concat()), feed_dict = m.get_feed_dict())

        q_sqrt_ref = np.ones((20,1))
        q_sqrt_ref2 = np.vstack(q_list)
        self.assertTrue(np.allclose(q_sqrt_ref, qsqrt))
        self.assertTrue(np.allclose(q_sqrt_ref2, qsqrt2))
        self.assertTrue(np.allclose(q_sqrt_ref, m.sqrtParam.concat()))
        self.assertTrue(np.allclose(q_sqrt_ref2, m.sqrtParam2.concat()))



if __name__ == "__main__":
    unittest.main()
