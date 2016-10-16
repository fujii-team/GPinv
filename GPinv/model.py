import tensorflow as tf
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import sys
from types import MethodType
from GPflow.model import Model, ObjectiveWrapper
from GPflow._settings import settings
from GPflow.param import AutoFlow
from .param import Param, ParamList, DataHolder, HierarchicParameterized, set_local_methods
float_type = settings.dtypes.float_type

class StVmodel(Model):
    """
    Base model for Stochastic Variational inference.
    """
    def __init__(self, kern, likelihood, mean_function, name='mode'):
        Model.__init__(self, name)
        self.kern = kern
        self.likelihood = likelihood
        self.mean_function = mean_function

    def build_predict(self):
        raise NotImplementedError

    def _sample(self, n_sample):
        """
        :param integer N: number of samples
        :Returns
         samples picked from the variational posterior.
         The Kulback_leibler divergence is stored as self._KL
        """
        raise NotImplementedError

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self.build_predict(Xnew)

    @AutoFlow((tf.int32, []))
    def sample_from_GP(self, n_sample):
        """
        Get samples from the posterior distribution.
        """
        return self._sample(n_sample[0])

    def sample_from(self, func_name, n_sample):
        """
        Sample from likelihood function.
        - n_samples integer: number of samples.
        - func_name string:  function name in likelihood.
        """
        method_name = 'sample_from_'+func_name
        # If this method does not have this method, we define and append it.
        if not hasattr(self, method_name):
            # A AutoFlowed method.
            def _build_sample_from_(self, n_sample):
                # generate samples from GP
                f_samples = self._sample(n_sample[0])
                # pass these samples to the likelihood method
                func = getattr(self.likelihood, func_name)
                return func(f_samples)
            # Append this method to self.
            setattr(self, method_name, _build_sample_from_)
            # Overwrite __name__ instance for method_name method
            method = getattr(self, method_name)
            method.__name__ = method_name            
        # Then, call this method.
        autoflow = AutoFlow((tf.int32, []))
        autoflow_runnable = autoflow(getattr(self, method_name))
        return autoflow_runnable(self, n_sample)


# TODO. Our model do not use np.optimize, providing much cleaner codes.
class HierarchicModel(Model):
    """
    This class provides two optimizers, grobal and local optimizer.
    In the local optimizer, only the localParams are optimized, while all the
    Params are optimized in the groval optimizer.
    """
    def __init__(self, name='model'):
        Model.__init__(self, name)
        # append local methods.
        set_local_methods(self, self)
        self._local_trainer = None
        self._global_trainer = None

    def _compile(self, global_trainer=None, local_trainer=None):
        """
        Overwrite Model._compile to deal with local and global optimizer, as
        well as to avoid recompilation for optimize_tf
        """
        # get local and global variables
        self._free_global_vars = tf.Variable(self.get_global_free_state())
        self._free_local_vars  = tf.Variable(self.get_local_free_state())
        # make tf_arrays from free_local_vars
        self.make_global_tf_array(self._free_global_vars)
        self.make_local_tf_array( self._free_local_vars)
        with self.tf_mode():
            f = self.build_likelihood() + self.build_prior()
            g_global, = tf.gradients(f, self._free_global_vars)
            g_local,  = tf.gradients(f, self._free_local_vars)
        self._minusF = tf.neg(f, name='objective')
        self._minusG_global = tf.neg(g_global, name='grad_objective')
        self._minusG_local  = tf.neg(g_local, name='grad_local_objective')
        # The optimiser needs to be part of the computational graph, and needs
        # to be initialised before tf.initialise_all_variables() is called.
        if global_trainer is None:
            self._train_global_op = None
        else:
            self._train_global_op = global_trainer.minimize(self._minusF,
                          var_list=[self._free_local_vars, self._free_global_vars])
            self._global_trainer = global_trainer
        if local_trainer is None:
            self._train_local_op = None
        else:
            self._train_local_op = local_trainer.minimize(self._minusF,
                                          var_list=[self._free_local_vars])
            self._local_trainer = local_trainer

        init = tf.initialize_all_variables()
        self._session.run(init)
        # build tensorflow functions for computing the likelihood
        if settings.verbosity.tf_compile_verb:
            print("compiling tensorflow local function...")
        sys.stdout.flush()

        # Methoe that to be minimized.
        def obj(x):
            global_size = len(self.get_global_free_state())
            x_global= x[:global_size]
            x_local = x[global_size:]
            feed_dict = {self._free_global_vars: x_global,
                         self._free_local_vars : x_local}
            feed_dict.update(self.get_feed_dict())
            f, g_global, g_local = self._session.run([self._minusF,
                                    self._minusG_global, self._minusG_local],
                                    feed_dict=feed_dict)
            return f.astype(np.float64), \
                np.hstack([g_global.astype(np.float64), g_local.astype(np.float64)])

        # Methoe that to be minimized.
        def obj_local(x):
            feed_dict = {self._free_local_vars: x}
            feed_dict.update(self.get_local_feed_dict())
            f, g = self._session.run([self._minusF, self._minusG_local],
                                     feed_dict=feed_dict)
            return f.astype(np.float64), g.astype(np.float64)

        self._objective = obj
        self._local_objective = obj_local
        # finish compilation.
        if settings.verbosity.tf_compile_verb:
            print("done")
        sys.stdout.flush()
        self._needs_recompile = False


    def optimize(self, method='L-BFGS-B', tol=None, callback=None,
                 maxiter=1000, **kw):
        """
        Optimize local and global parameters.
        """
        if type(method) is str:
            return self._optimize_np(method, tol, callback, False, maxiter, **kw)
        else:
            return self._optimize_global_tf(method, callback, maxiter)

    def optimize_local(self, method='L-BFGS-B', tol=None, callback=None,
                 maxiter=1000, **kw):
        """
        Optimize local parameters.
        """
        if type(method) is str:
            return self._optimize_np(method, tol, callback, True, maxiter, **kw)
        else:
            return self._optimize_local_tf(method, callback, maxiter)

    def _set_free_global_vars(self):
        """
        Set the current np.value to _free_global_vars
        """
        assign_global=self._free_global_vars.assign(self.get_global_free_state())
        assign_local =self._free_local_vars.assign(self.get_local_free_state())
        self._session.run([assign_local, assign_global])

    def _set_free_local_vars(self):
        """
        Set the current np.value to _free_local_vars
        """
        assign_local=self._free_local_vars.assign(self.get_local_free_state())
        self._session.run(assign_local)

    def copy_trainer_local_to_global(self):
        """
        Copy auxiliary variables in _local_trainer to _global_trainer.
        """
        if hasattr(self, '_global_trainer') and \
           hasattr(self, '_local_trainer'):
            slots_names_global = self._global_trainer.get_slot_names()
            slots_names_local = self._local_trainer.get_slot_names()
            slot_var_global_list = []
            for sname in slots_names_global:
                if sname in slots_names_local:
                    slot_var_local  = self._local_trainer.get_slot(self._free_local_vars, sname)
                    slot_var_global = self._global_trainer.get_slot(self._free_local_vars, sname)
                    if slot_var_local is not None and slot_var_global is not None:
                        slot_var_global.assign(slot_var_local)
                        slot_var_global_list.append(slot_var_global)
            self._session.run(tf.initialize_variables(slot_var_global_list))


    def _optimize_np(self, method='L-BFGS-B', tol=None, callback=None,
                    local = False, maxiter=1000, **kw):
        """
        Overwrite Model._optimize_np with local or global optimization flag.
        """
        # Recompile if necessary.
        # TODO to also compile for the tf-version optimizer
        # if self._method_local is available
        if self._needs_recompile:
            self._compile()
        options = dict(disp=settings.verbosity.optimisation_verb, maxiter=maxiter)
        options.update(kw)
        # here's the actual call to minimize. Catch keyboard errors as harmless.
        try:
            if not local:
                obj = ObjectiveWrapper(self._objective)
                result = minimize(fun=obj,
                              x0=np.hstack([self.get_global_free_state(),self.get_local_free_state()]),
                              method=method,
                              jac=True,
                              tol=tol,
                              callback=callback,
                              options=options)
            else:
                obj = ObjectiveWrapper(self._local_objective)
                result = minimize(fun=obj,
                              x0=self.get_local_free_state(),
                              method=method,
                              jac=True,
                              tol=tol,
                              callback=callback,
                              options=options)

        # If caught KeyboardInterrupt
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting \
                  model with most recent state.")
            if not local:
                count = self.set_global_state(obj._previous_x)
                self.set_local_state( obj._previous_x[count:])
                self._set_free_global_vars()
            else:
                self.set_local_state(obj._previous_x)
                self._set_free_local_vars()
            return None
        # Optimize finished
        if settings.verbosity.optimisation_verb:
            print("optimization terminated, setting model state")
        if not local:
            count = self.set_global_state(obj._previous_x)
            self.set_local_state( obj._previous_x[count:])
            self._set_free_global_vars()
        else:
            self.set_local_state(result.x)
            self._set_free_local_vars()
        return result


    def _optimize_global_tf(self, method, callback, maxiter):
        if self._needs_recompile or not hasattr(self, '_train_global_op') \
                or self._global_trainer is not method:
            self._compile(global_trainer=method,
                          local_trainer= self._local_trainer)
            self._global_trainer = method
        try:
            iteration = 0
            while iteration < maxiter:
                self._session.run(self._train_global_op,
                    feed_dict=self.get_feed_dict())
                if callback is not None:
                    callback(np.hstack(self._session.run(
                            [self._free_global_vars, self._free_local_vars])))
                iteration += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting model\
                  with most recent state.")
            self.set_global_state(self._session.run(self._free_global_vars))
            self.set_local_state( self._session.run(self._free_local_vars))
            return None

        final_x_global = self._session.run(self._free_global_vars)
        final_x_local  = self._session.run(self._free_local_vars)
        self.set_global_state(final_x_global)
        self.set_local_state(final_x_local)
        final_x = np.hstack([final_x_global, final_x_local])
        fun, jac = self._objective(final_x)
        r = OptimizeResult(x=final_x,
                           success=True,
                           message="Finished iterations.",
                           fun=fun,
                           jac=jac,
                           status="Finished iterations.")
        return r

    def _optimize_local_tf(self, method, callback, maxiter):
        if self._needs_recompile or not hasattr(self, '_train_local_op') \
                or self._local_trainer is not method:
            self._compile(global_trainer=self._global_trainer,
                          local_trainer =method)
            self._local_trainer = method
        try:
            iteration = 0
            while iteration < maxiter:
                self._session.run(self._train_local_op,
                    feed_dict=self.get_local_feed_dict())
                if callback is not None:
                    callback(np.hstack(self._session.run(self._free_local_vars)))
                iteration += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting model\
                  with most recent state.")
            self.set_local_state( self._session.run(self._free_local_vars))
            return None

        final_x = self._session.run(self._free_local_vars)
        self.set_local_state(final_x)
        fun, jac = self._local_objective(final_x)
        r = OptimizeResult(x=final_x,
                           success=True,
                           message="Finished iterations.",
                           fun=fun,
                           jac=jac,
                           status="Finished iterations.")
        return r
