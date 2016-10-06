from __future__ import print_function
import tensorflow as tf
import numpy as np
import unittest
import GPinv
import GPflow

class Quadratic(GPinv.model.HierarchicModel):
    def __init__(self):
        GPinv.model.HierarchicModel.__init__(self)
        self.rng = np.random.RandomState(0)
        self.x = GPinv.param.LocalParam(self.rng.randn(2,1))
        self.y = GPflow.param.Param(self.rng.randn(2,1))
        self.m = GPflow.param.Parameterized()
        self.m.z = GPflow.param.Param(self.rng.randn(2,1))
        self.m.x= GPinv.param.LocalParam(self.rng.randn(2,1))

    def build_likelihood(self):
        return -tf.reduce_sum(tf.square(self.x - np.ones((2,1))) + \
                              tf.square(self.y - np.ones((2,1))*0.1)+ \
                              tf.square(self.m.z - np.ones((2,1))*0.01) +\
                              tf.square(self.m.x - np.ones((2,1))*0.001))


class test_optimize(unittest.TestCase):
    def setUp(self):
        self.m = Quadratic()

    def test_optimize_global(self):
        self.m.optimize()
        x = self.m.x.value
        y = self.m.y.value
        self.assertTrue(np.allclose(x, np.ones((2,1)), rtol=1.0e-5))
        self.assertTrue(np.allclose(y, np.ones((2,1))*0.1, rtol=1.0e-5))

    def test_optimize_local(self):
        rslt = self.m.optimize_local()
        # store logp method
        logp_local = self.m._local_objective(self.m.get_local_free_state())[0]
        x = self.m.x.value
        y = self.m.y.value
        self.assertTrue(np.allclose(x, np.ones((2,1)), rtol=1.0e-5))
        self.assertFalse(np.allclose(y, np.ones((2,1))*0.1, rtol=1.0e-5))
        # Recompile
        self.m._needs_recompile = True
        rslt = self.m.optimize()
        rslt = self.m.optimize_local(maxiter=1)
        # Assert after the global optimization, logp value changes.
        logp_local2  = self.m._local_objective(self.m.get_local_free_state())[0]
        self.assertFalse(np.allclose(logp_local, logp_local2))
        # assert the _free_local_vars is updated.

    def test_optimize_local_free_vars(self):
        """ Make sure _free_local_vars are updated by optimize_local() """
        # compile and store the current _free_local_vars
        rslt = self.m._compile()
        local_var0 = self.m._session.run(self.m._free_local_vars)
        # optimize the updated _free_local_vars
        rslt = self.m.optimize_local()
        local_var1 = self.m._session.run(self.m._free_local_vars)
        local_var_np = self.m.get_local_free_state()
        # make sure _free_local_vars are updated
        self.assertTrue( np.allclose(local_var1, local_var_np))
        self.assertFalse(np.allclose(local_var1, local_var0))


    def test_optimize_tf(self):
        method = tf.train.AdamOptimizer(learning_rate=0.01)
        rslt = self.m.optimize(method)
        x = self.m.x.value
        y = self.m.y.value
        self.assertTrue(np.allclose(x, np.ones((2,1)), rtol=1.0e-5))
        self.assertTrue(np.allclose(y, np.ones((2,1))*0.1, rtol=1.0e-5))
        # assert if compilation is avoided when the same optimizer used.
        rslt = self.m.optimize(method, maxiter=1)
        self.assertTrue(method is self.m._global_trainer)
        # assert compilation if the optimizer was changed.
        method2 = tf.train.AdadeltaOptimizer(learning_rate=0.01)
        rslt = self.m.optimize(method2, maxiter=1)
        self.assertFalse(method is self.m._global_trainer)
        # assert work after the np.optimize
        rslt = self.m.optimize()
        rslt = self.m.optimize(method2, maxiter=1)
        self.assertFalse(method is self.m._global_trainer)

    def test_optimize_local_tf(self):
        method_global = tf.train.AdamOptimizer(learning_rate=0.01)
        method_local = tf.train.AdamOptimizer(learning_rate=0.01)
        self.m.optimize(method_global, maxiter=1)
        obj_global = self.m._objective(np.hstack(
                    [self.m.get_global_free_state(),self.m.get_local_free_state()]))[0]
        self.m.optimize_local(method_local)
        x = self.m.x.value
        y = self.m.y.value
        obj_local   = self.m._local_objective(self.m.get_local_free_state())[0]
        obj_global2 = self.m._objective(np.hstack(
                    [self.m.get_global_free_state(),self.m.get_local_free_state()]))[0]
        print(obj_global, obj_local, obj_global2)
        self.assertTrue(np.allclose(x, np.ones((2,1)), rtol=1.0e-4))
        self.assertFalse(np.allclose(y, np.ones((2,1))*0.1, rtol=1.0e-5))
        # make sure the objective decreases by local optimizer
        self.assertTrue(obj_local   < obj_global)
        self.assertTrue(obj_global2 < obj_global)
        # make sure the _objective and _local_objective is the same
        self.assertTrue(obj_global2 == obj_local)
        # assert if compilation is avoided when the same optimizer used.
        rslt = self.m.optimize_local(method_local, maxiter=1)
        self.assertTrue(method_local is self.m._local_trainer)
        # assert compilation if the optimizer was changed.
        method2 = tf.train.AdadeltaOptimizer(learning_rate=0.01)
        rslt = self.m.optimize_local(method2, maxiter=1)
        self.assertFalse(method_local is self.m._local_trainer)

    def test_avoid_duplicate_compilation(self):
        method_global = tf.train.AdamOptimizer(learning_rate=0.01)
        method_local  = tf.train.AdamOptimizer(learning_rate=0.01)
        # Both local global optimizer should be initialized.
        self.m._compile(global_trainer=method_global, local_trainer=method_local)
        # make sure needs_recompile flag is not raised.
        self.assertFalse(self.m._needs_recompile)
        # self.m should have _train_global_op and _train_local_op
        self.assertTrue(self.m._train_global_op is not None)
        self.assertTrue(self.m._train_local_op is not None)
        # make sure the slots are not Zero
        self.m.optimize(method_global, maxiter=10)
        self.m.optimize_local(method_local, maxiter=1)
        gslot_local = self.m._session.run(method_global.get_slot(self.m._free_local_vars, 'm'))
        self.assertFalse(np.allclose(gslot_local, np.zeros(4)))


    def test_copy_slots(self):
        """ make sure the slots are copied from local method to global method """
        method_global = tf.train.AdamOptimizer(learning_rate=0.01)
        method_local  = tf.train.AdamOptimizer(learning_rate=0.01)
        self.m._compile(global_trainer=method_global, local_trainer=method_local)
        self.m.optimize(method_global, maxiter=10)
        self.m.optimize_local(method_local, maxiter=1)
        slots_names = method_global.get_slot_names()
        gslot_local = self.m._session.run(method_global.get_slot(self.m._free_local_vars, 'm'))
        gslot_global= self.m._session.run(method_global.get_slot(self.m._free_global_vars, 'm'))
        lslot_local = self.m._session.run(method_local.get_slot(self.m._free_local_vars, 'm'))
        self.m.copy_trainer_local_to_global()
        gslot_local2 = self.m._session.run(method_global.get_slot(self.m._free_local_vars, 'm'))
        gslot_global2= self.m._session.run(method_global.get_slot(self.m._free_global_vars, 'm'))
        lslot_local2 = self.m._session.run(method_local.get_slot(self.m._free_local_vars, 'm'))
        # auxiliary variables for LocalParam in the global trainer should change
        self.assertFalse(np.allclose(gslot_local, gslot_local2))
        # auxiliary variables for GlobalParam in the global trainer should NOT change
        self.assertTrue(np.allclose(gslot_global, gslot_global))
        # auxiliary variables for LocalParam in the local trainer should NOT change
        self.assertTrue(np.allclose(lslot_local, lslot_local2))


    def test_local_train_var(self):
        method_global = tf.train.AdamOptimizer(learning_rate=0.01)
        method_local  = tf.train.AdamOptimizer(learning_rate=0.01)
        self.m.optimize(method_global, maxiter=10)
        gslot_local = self.m._session.run(method_global.get_slot(self.m._free_local_vars, 'm'))
        self.m.set_local_train_var(gslot_local, 'm')
        # make sure the train_var is stored LocalParam._train_var
        print(self.m.x._train_var)
        self.assertTrue('m' in self.m.x._train_var.keys())
        # make sure the local_train_var is restored by get_local_train_var
        gslot_local2 = self.m.get_local_train_var('m')
        print(gslot_local2)
        self.assertTrue(np.allclose(gslot_local, gslot_local2))

    def test_local_train_var_fixed(self):
        method_global = tf.train.AdamOptimizer(learning_rate=0.01)
        method_local  = tf.train.AdamOptimizer(learning_rate=0.01)
        self.m.x.fixed=True
        self.m.optimize(method_global, maxiter=10)
        gslot_local = self.m._session.run(method_global.get_slot(self.m._free_local_vars, 'm'))
        self.m.set_local_train_var(gslot_local, 'm')
        # make sure the train_var is stored LocalParam._train_var
        print(self.m.x._train_var)
        self.assertTrue('m' in self.m.x._train_var.keys())
        # make sure the local_train_var is restored by get_local_train_var
        gslot_local2 = self.m.get_local_train_var('m')
        print(gslot_local2)
        self.assertTrue(np.allclose(gslot_local, gslot_local2))


class KeyboardRaiser:
    """
    This wraps a function and makes it raise a KeyboardInterrupt after some number of calls
    """
    def __init__(self, iters_to_raise, f):
        self.iters_to_raise, self.f = iters_to_raise, f
        self.count = 0

    def __call__(self, *a, **kw):
        self.count += 1
        if self.count >= self.iters_to_raise:
            raise KeyboardInterrupt
        return self.f(*a, **kw)

class TestKeyboardCatching(unittest.TestCase):
    def setUp(self):
        self.m = Quadratic()

    def test_optimize_np(self):
        x0 = self.m.get_free_state()
        self.m._compile()
        self.m._objective = KeyboardRaiser(5, self.m._objective)
        self.m.optimize(disp=0, maxiter=10000, ftol=0, gtol=0)
        x1 = self.m.get_free_state()
        self.assertFalse(np.allclose(x0, x1))

    def test_optimize_tf(self):
        x0 = self.m.get_free_state()
        callback = KeyboardRaiser(5, lambda x: None)
        o = tf.train.AdamOptimizer()
        self.m.optimize(o, maxiter=15, callback=callback)
        x1 = self.m.get_free_state()
        self.assertFalse(np.allclose(x0, x1))

    def test_optimize_local_np(self):
        x0 = self.m.get_local_free_state()
        self.m._compile()
        self.m._objective = KeyboardRaiser(5, self.m._objective)
        self.m.optimize_local(disp=0, maxiter=10000, ftol=0, gtol=0)
        x1 = self.m.get_local_free_state()
        self.assertFalse(np.allclose(x0, x1))

    def test_optimize_local_tf(self):
        x0 = self.m.get_local_free_state()
        callback = KeyboardRaiser(5, lambda x: None)
        o = tf.train.AdamOptimizer()
        self.m.optimize_local(o, maxiter=15, callback=callback)
        x1 = self.m.get_local_free_state()
        self.assertFalse(np.allclose(x0, x1))


if __name__ == '__main__':
    unittest.main()
