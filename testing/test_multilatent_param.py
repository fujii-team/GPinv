import tensorflow as tf
import GPflow
import GPinv
from GPinv import transforms
from GPinv.multilatent_param import ModelInput, IndexedDataHolder, IndexedParamList, ConcatParamList, SqrtParamList
import numpy as np
import unittest

class test_IndexedDataHolder_ParamList_SqrtParamList(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X1 = self.rng.randn(10, 2)
        self.Z1 = self.rng.randn( 3, 2)
        self.model_input1 = ModelInput(self.X1, GPinv.kernels.RBF(2), self.Z1,
                        X_minibatch_size=3, random_seed=1)

        self.X2 = self.rng.randn(10, 2)
        self.Z2 = self.rng.randn( 3, 2)
        self.model_input2 = ModelInput(self.X2, GPinv.kernels.RBF(2), self.Z2,
                        X_minibatch_size=3, random_seed=2)
        # manual input of the reference
        ref = []
        for x in self.X1:
            ref.append([x[0], x[1], 0])
        for x in self.X2:
            ref.append([x[0], x[1], 1])
        self.ref = np.array(ref)

        refZ = []
        for z in self.Z1:
            refZ.append([z[0], z[1], 0])
        for z in self.Z2:
            refZ.append([z[0], z[1], 1])
        self.refZ = np.array(refZ)

    def test_IndexedDataHolder(self):
        m = GPflow.model.Model()
        m.indexedData = IndexedDataHolder([self.model_input1, self.model_input2])
        with m.tf_mode():
            X = m._session.run(m.indexedData, feed_dict = m.get_feed_dict())
        self.assertTrue(np.allclose(self.ref, X))
        # make sure the last dimension is the index
        self.assertTrue(X[2, -1] == 0.0)
        self.assertTrue(X[12, -1] == 1.0)

        # test getitem
        self.assertTrue(np.allclose(self.X1, m.indexedData[0]))
        # test setitem
        X2 = self.rng.randn(10,2)
        m.indexedData[1] = X2
        self.assertTrue(np.allclose(X2, m.indexedData[1]))

    def test_IndexedParamList(self):
        m = GPflow.model.Model()
        m.indexedParam = IndexedParamList([self.model_input1, self.model_input2])
        self.assertTrue(np.allclose(self.refZ, m.indexedParam.concat()))
        # tf_mode
        free_var = m.get_free_state()
        m.make_tf_array(free_var)
        with m.tf_mode():
            Z = m._session.run(m.indexedParam.concat(), feed_dict = m.get_feed_dict())
        self.assertTrue(np.allclose(self.refZ, Z))

    def test_SqrtParamList(self):
        m = GPflow.model.Model()
        indices_list = [(0,0),(1,1),(1,0)]
        num_latent = 2
        m.sqrtParamList = SqrtParamList([self.model_input1, self.model_input2],
                                        num_latent,
                                        q_shape='specified',
                                        indices_list=indices_list)
        # check without tf_mode
        ref_sqrt = np.tile(np.expand_dims(np.eye(self.Z1.shape[0] + self.Z2.shape[0]),
                                        2), [1,1,num_latent])
        self.assertTrue(np.allclose(ref_sqrt, m.sqrtParamList.concat()))
        # check with tf_mode
        # tf_mode
        free_var = m.get_free_state()
        m.make_tf_array(free_var)
        with m.tf_mode():
            sqrt = m._session.run(m.sqrtParamList.concat(), feed_dict = m.get_feed_dict())
        self.assertTrue(np.allclose(ref_sqrt, m.sqrtParamList.concat()))
        # check no-good index
        with self.assertRaises(ValueError):
            tmp = SqrtParamList([self.model_input1, self.model_input2],
                                         num_latent,
                                         q_shape='specified',
                                         indices_list=[(0,0),(1,1),(0,1)])

    def test_SqrtParamList_given(self):
        m = GPflow.model.Model()
        indices_list = [(0,0),(1,1),(1,0)]
        num_latent = 2
        sqrt_input =[]
        for i in range(3):
            sqrt_input.append(self.rng.randn(3,3,2))
        # manual creation of the reference.
        ref_sqrt = np.pad(sqrt_input[0],[[0,3],[0,3],[0,0]], mode='constant')
        ref_sqrt+= np.pad(sqrt_input[1],[[3,0],[3,0],[0,0]], mode='constant')
        ref_sqrt+= np.pad(sqrt_input[2],[[3,0],[0,3],[0,0]], mode='constant')

        m.sqrtParamList = SqrtParamList([self.model_input1, self.model_input2],
                            num_latent,
                            q_shape='specified',
                            indices_list=indices_list,
                            q_sqrt_list=sqrt_input)
        # check without tf_mode
        self.assertTrue(np.allclose(ref_sqrt, m.sqrtParamList.concat()))

    def test_SqrtParamList_full(self):
        m = GPflow.model.Model()
        num_latent = 2
        m.sqrtParamList = SqrtParamList([self.model_input1, self.model_input2],
                            num_latent,
                            q_shape='fullrank')
        # check without tf_mode
        ref_sqrt = np.tile(np.expand_dims(np.eye(self.Z1.shape[0] + self.Z2.shape[0]),
                                        2), [1,1,num_latent])
        # check without tf_mode
        self.assertTrue(np.allclose(ref_sqrt, m.sqrtParamList.concat()))

    def test_ConcatParamList(self):
        m = GPflow.model.Model()
        num_latent = 2
        m.diagParamList = ConcatParamList([self.model_input1, self.model_input2],
                            num_latent)
        # check without tf_mode
        ref_diag = np.ones((self.Z1.shape[0] + self.Z2.shape[0], num_latent))
        # check without tf_modek
        self.assertTrue(np.allclose(ref_diag, m.diagParamList.concat()))

    def test_ConcatParamList_given(self):
        m = GPflow.model.Model()
        num_latent = 2
        diag_input = []
        for i in range(2):
            diag_input.append(self.rng.randn(3,num_latent))
        m.diagParamList = ConcatParamList([self.model_input1, self.model_input2],
                            num_latent, value_list = diag_input)

        # check without tf_mode
        ref_diag = np.vstack(diag_input)
        # check without tf_modek
        self.assertTrue(np.allclose(ref_diag, m.diagParamList.concat()))



if __name__ == "__main__":
    unittest.main()
