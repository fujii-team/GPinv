from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import GPinv
import GPflow

class test_Param(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.m = GPinv.param.HierarchicParameterized()
        self.m.m = GPflow.param.Parameterized()
        # local params
        self.local_p1 = rng.randn(10,2)
        self.local_p2 = rng.randn(11,2)
        self.m.local_p1 = GPinv.param.LocalParam(self.local_p1.copy())
        self.m.m.local_p2 = GPinv.param.LocalParam(self.local_p2.copy())
        self.global_p1 = rng.randn(13,1)
        self.global_p2 = rng.randn(14,1)
        self.m.global_p1 = GPflow.param.Param(self.global_p1.copy())
        self.m.m.global_p2 = GPflow.param.Param(self.global_p2.copy())
        self.d1 = rng.randn(15,1)
        self.d2 = rng.randn(16,1)
        self.m.d1 = GPflow.param.DataHolder(self.d1.copy())
        self.m.m.d2 = GPflow.param.DataHolder(self.d2.copy())


    def test_local_free_state(self):
        self.assertTrue(np.allclose(self.m.get_local_free_state(),
                        np.hstack([self.local_p1.flatten(),
                                   self.local_p2.flatten()])) or
                        np.allclose(self.m.get_local_free_state(),
                        np.hstack([self.local_p2.flatten(),
                                   self.local_p1.flatten()])))

    def test_global_free_state(self):
        self.assertTrue(np.allclose(self.m.get_global_free_state(),
                        np.hstack([self.global_p1.flatten(),
                                   self.global_p2.flatten()])) or
                        np.allclose(self.m.get_global_free_state(),
                        np.hstack([self.global_p2.flatten(),
                                   self.global_p1.flatten()])))

    def test_local_feed_dict(self):
        self.m.make_tf_array(self.m.get_free_state())
        local_feed_dict = self.m.get_local_feed_dict()
        # check if global parameter is contained in feed_dict
        self.assertTrue(self.m.global_p1._tf_array in local_feed_dict.keys())
        self.assertTrue(self.m.m.global_p2._tf_array in local_feed_dict.keys())
        # check if local parameter is NOT contained in feed_dict
        self.assertFalse(self.m.local_p1._tf_array in local_feed_dict.keys())
        self.assertFalse(self.m.m.local_p2._tf_array in local_feed_dict.keys())
        # check if dataholder is contained in feed_dict
        self.assertTrue(self.m.d1._tf_array in local_feed_dict.keys())
        self.assertTrue(self.m.m.d2._tf_array in local_feed_dict.keys())
        self.assertTrue(np.allclose(
            local_feed_dict[self.m.d1._tf_array], self.d1))
        self.assertTrue(np.allclose(
            local_feed_dict[self.m.m.d2._tf_array], self.d2))

    """
    global_feed_dict is deprecated
    def test_global_feed_dict(self):
        self.m.make_tf_array(self.m.get_free_state())
        global_feed_dict = self.m.get_global_feed_dict()
        # check if global parameter is NOT contained in feed_dict
        self.assertFalse(self.m.global_p1._tf_array in global_feed_dict.keys())
        self.assertFalse(self.m.m.global_p2._tf_array in global_feed_dict.keys())
        # check if local parameter is contained in feed_dict
        self.assertTrue(self.m.local_p1._tf_array in global_feed_dict.keys())
        self.assertTrue(self.m.m.local_p2._tf_array in global_feed_dict.keys())
        # check if dataholder is contained in feed_dict
        self.assertTrue(self.m.d1._tf_array in global_feed_dict.keys())
        self.assertTrue(self.m.m.d2._tf_array in global_feed_dict.keys())
        self.assertTrue(np.allclose(
            global_feed_dict[self.m.d1._tf_array], self.d1))
        self.assertTrue(np.allclose(
            global_feed_dict[self.m.m.d2._tf_array], self.d2))
    """

    def test_set_local_state(self):
        rng = np.random.RandomState(1)
        local_state = self.m.get_local_free_state()
        self.m.set_local_state(rng.randn(local_state.shape[0]))
        # local parameters should be replaced
        self.assertFalse(np.allclose(self.m.local_p1.value, self.local_p1))
        self.assertFalse(np.allclose(self.m.m.local_p2.value, self.local_p2))
        # global parameters should not be changed
        self.assertTrue(np.allclose(self.m.global_p1.value, self.global_p1))
        self.assertTrue(np.allclose(self.m.m.global_p2.value, self.global_p2))

    def test_set_global_state(self):
        rng = np.random.RandomState(1)
        global_state = self.m.get_global_free_state()
        self.m.set_global_state(rng.randn(global_state.shape[0]))
        # local parameters should not be replaced
        self.assertTrue(np.allclose(self.m.local_p1.value, self.local_p1))
        self.assertTrue(np.allclose(self.m.m.local_p2.value, self.local_p2))
        # global parameters should not be changed
        self.assertFalse(np.allclose(self.m.global_p1.value, self.global_p1))
        self.assertFalse(np.allclose(self.m.m.global_p2.value, self.global_p2))

    def test_get_local_params(self):
        local_params = self.m.get_local_params()
        self.assertTrue(self.m.local_p1 in local_params)
        self.assertTrue(self.m.m.local_p2 in local_params)
        self.assertFalse(self.m.global_p1 in local_params)
        self.assertFalse(self.m.m.global_p2 in local_params)


class test_DataSet(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.m = GPinv.param.HierarchicParameterized()
        self.m.m = GPflow.param.Parameterized()
        #LocalDataManager
        self.d = []
        for i in range(3):
            self.d.append({'x':rng.randn(15,2), 'y':rng.randn(16,2)})
        self.m.manager = \
         GPinv.param.LocalDataManager(self.d, minibatch_size=2, random_seed=0)
        self.m.d1 =self.m.manager.data_holders['x']
        self.m.m.d2 =self.m.manager.data_holders['y']
        # local params
        self.local_p1 = rng.randn(10,2)
        self.local_p2 = rng.randn(11,2)
        self.m.local_p1 = GPinv.param.LocalParam(self.local_p1.copy(),
                        transform=GPflow.transforms.positive)
        self.m.m.local_p2 = GPinv.param.LocalParam(self.local_p2.copy())
        self.global_p1 = rng.randn(13,1)
        self.global_p2 = rng.randn(14,1)
        self.m.global_p1 = GPflow.param.Param(self.global_p1.copy())
        self.m.m.global_p2 = GPflow.param.Param(self.global_p2.copy())

    def test_store_local_param(self):
        self.m.manager._store_local_param()
        for i in range(2):
            self.assertTrue(self.m.name+'.local_p1' in self.m.manager.current_set[i].keys())
            self.assertTrue(self.m.name+'.m.local_p2' in self.m.manager.current_set[i].keys())
            self.assertTrue(np.allclose(
                self.m.manager.current_set[i][self.m.name+'.local_p1']['value'],
                self.m.local_p1.value[i]))
            self.assertTrue(np.allclose(
                self.m.manager.current_set[i][self.m.name+'.m.local_p2']['value'],
                self.m.m.local_p2.value[i]))

    def test_restore_local_param(self):
        self.m.manager.next_set()
        # count the number of stored parameter
        count_p1 = 0
        count_p2 = 0
        for d in self.m.manager.data_set:
            if self.m.name+'.local_p1' in d.keys():
                count_p1 += 1
                count_p2 += 1
                # assert one of local param is stored.
                self.assertTrue(any([np.allclose(d[self.m.name+'.local_p1']['value'], p)
                                        for p in self.local_p1]))
                self.assertTrue(any([np.allclose(d[self.m.name+'.m.local_p2']['value'], p)
                                        for p in self.local_p2]))
        self.assertTrue(count_p1 is 2)
        self.assertTrue(count_p2 is 2)
        # Test data holder
        x = self.m.manager['x'].value
        y = self.m.manager['y'].value
        self.assertTrue(any([np.allclose(d['x'], x[0]) for d in self.d]))
        self.assertTrue(any([np.allclose(d['y'], y[0]) for d in self.d]))
        self.assertTrue(any([np.allclose(d['x'], x[1]) for d in self.d]))
        self.assertTrue(any([np.allclose(d['y'], y[1]) for d in self.d]))


if __name__ == '__main__':
    unittest.main()
