import tensorflow as tf
import numpy as np
import types
import sys
from GPflow import param
#from GPflow.param import Parentable, Param, Parameterized, DataHolder
from GPflow.scoping import NameScoped
from GPflow import transforms
from GPflow._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Param(param.Param):
    pass

class DataHolder(param.DataHolder):
    pass

class Parameterized(param.Parameterized):
    pass

class ParamList(param.ParamList):
    pass

"""
Here, we define some methods to be appended to Param and Parameterized instances
to control local parameters.
"""
def set_local_methods(self, child):
    """
    set some local methods.
    """
    # for Param instance
    if isinstance(child, param.Param):
        def _get_local_feed_dict(self):
            return {self._tf_array: self.value}
        child.get_local_feed_dict = types.MethodType(_get_local_feed_dict, child)
        # add get_global_free_state method only for Param (not for LocalParam)
        if not isinstance(child, LocalParam):
            def _get_global_free_state(self):
                return self.get_free_state()
            child.get_global_free_state = types.MethodType(_get_global_free_state, child)

    # for Parameterized instance.
    elif isinstance(child, param.Parameterized):
        # Append this 'set_local_methods' first.
        child.set_local_methods = types.MethodType(set_local_methods, child)
        # Next append some local methods.
        child.get_local_feed_dict = types.MethodType(get_local_feed_dict, child)
        # get local and global free_state
        child.get_local_free_state = types.MethodType(get_local_free_state, child)
        child.get_global_free_state = types.MethodType(get_global_free_state, child)
        # set_local_state and set_global_state
        child.set_local_state = types.MethodType(set_local_state, child)
        child.set_global_state = types.MethodType(set_global_state, child)
        # make_local_tf_array, make_global_tf_array
        child.make_local_tf_array = types.MethodType(make_local_tf_array, child)
        child.make_global_tf_array = types.MethodType(make_global_tf_array, child)
        # get_local_params
        child.get_local_params = types.MethodType(get_local_params, child)
        # set_local_data
        #child.get_local_data = types.MethodType(get_local_data, child)
        # get_local_auxil_var
        child.set_local_train_var = types.MethodType(set_local_train_var, child)
        child.get_local_train_var = types.MethodType(get_local_train_var, child)

def get_local_feed_dict(self):
    """
    Recursively fetch a dictionary matching up placeholders for the
    fixed-LocalParam and for the global parameters.
    """
    d = {}
    for p in self.sorted_params + self.data_holders:
        # call usual get_feed_dict for LocalParam and DataHolder
        if isinstance(p, (param.DataHolder, LocalParam)):
            d.update(p.get_feed_dict())
        else:
            # for the legacy Param instance, we append get_local_feed_dict method.
            if not hasattr(p, "get_local_feed_dict"):
                self.set_local_methods(p)
            d.update(p.get_local_feed_dict())
    return d

def get_local_free_state(self):
    """
    Recurse get_local_free_state on all child parameters, and hstack them.
    Return: Stacked np-array for all LocalParam
    """
    # check if the child has 'get_local_free_state' method
    for p in self.sorted_params:
        if isinstance(p, (param.Param,param.Parameterized)) and \
        not hasattr(p, 'get_local_free_state'):
            self.set_local_methods(p)
    # Here, additional empty array allows hstacking of empty list
    return np.hstack([p.get_local_free_state() for p in self.sorted_params
                       if isinstance(p, (param.Parameterized,LocalParam))]
                     + [np.empty(0, np_float_type)])

def get_global_free_state(self):
    """
    Recurse get_global_free_state on all child parameters, and hstack them.
    Return: Stacked np-array for all Param except for LocalParam
    """
    # check if the child has 'get_local_free_state' method
    for p in self.sorted_params:
        if isinstance(p, (param.Param,param.Parameterized)) and \
        not hasattr(p, 'get_global_free_state'):
            self.set_local_methods(p)
    # Here, additional empty array allows hstacking of empty list
    return np.hstack([p.get_global_free_state() for p in self.sorted_params
                       if isinstance(p, (param.Parameterized, param.Param))]
                    + [np.empty(0, np_float_type)])

def set_local_state(self, x):
    """
    Set the values of all the local-parameters by recursion
    """
    count = 0
    for p in self.sorted_params:
        # Append method if necessary
        if isinstance(p, param.Parameterized) and not hasattr(p, 'set_local_state'):
            self.set_local_methods(p)
        # Start to set
        if isinstance(p, LocalParam): # Call set_state for local parameter
            count += p.set_state(x[count:])
        elif isinstance(p, param.Param): # Do nothing for global parameter
            pass
        else: # for parameterized
            count += p.set_local_state(x[count:])
    return count

def set_global_state(self, x):
    """
    Set the values of all the local-parameters by recursion
    """
    count = 0
    for p in self.sorted_params:
        # Append method if necessary
        if isinstance(p, param.Parameterized) and not hasattr(p, 'set_global_state'):
            self.set_local_methods(p)
        # Start to set
        if isinstance(p, LocalParam): # Do nothing for local parameter
            pass
        elif isinstance(p, param.Param): # Call set_state for global parameter
            count += p.set_state(x[count:])
        else: # for parameterized
            count += p.set_global_state(x[count:])
    return count

def make_local_tf_array(self, X):
    count = 0
    for p in self.sorted_params:
        # Append method if necessary
        if isinstance(p, param.Parameterized) and not hasattr(p, 'make_local_tf_array'):
            self.set_local_methods(p)
        # Start to set
        if isinstance(p, LocalParam): # Call make_tf_array for local parameter
            count += p.make_tf_array(X[count:])
        elif isinstance(p, param.Param): # Do nothing for global parameter
            pass
        else: # for parameterized
            count += p.make_local_tf_array(X[count:])
    return count

def make_global_tf_array(self, X):
    count = 0
    for p in self.sorted_params:
        # Append method if necessary
        if isinstance(p, param.Parameterized) and not hasattr(p, 'make_global_tf_array'):
            self.set_local_methods(p)
        # Start to set
        if isinstance(p, LocalParam): # Do nothing for global parameter
            pass
        elif isinstance(p, param.Param): # Call make_tf_array for local parameter
            count += p.make_tf_array(X[count:])
        else: # for parameterized
            count += p.make_global_tf_array(X[count:])
    return count

def get_local_params(self):
    """
    Get list of instances of LocalParam in this and child classes recursively.
    """
    local_params = []
    for p in self.sorted_params:
        # Append method if necessary
        if isinstance(p, param.Parameterized) and not hasattr(p, 'get_local_params'):
            self.set_local_methods(p)
        # look for local params
        if isinstance(p, LocalParam):
            local_params.append(p)
        elif isinstance(p, param.Parameterized):
            local_params=local_params+p.get_local_params()
    return local_params

'''
def get_local_data(self):
    """
    Get list of instances of LocalDataHolder in this and child classes recursively.
    """
    local_data = []
    for p in self.sorted_params + self.data_holders:
        # Append method if necessary
        if isinstance(p, param.Parameterized) and not hasattr(p, 'get_local_data'):
            self.set_local_methods(p)
        # look for local data
        if isinstance(p, LocalData):
            local_data.append(p)
        elif isinstance(p, param.Parameterized):
            local_data.append(p.get_local_data())
    return local_data
'''

def set_local_train_var(self, x, name):
    """
    Set the optimization-relavant values of all the local-parameters by
    recursion.
    Typically, this is used for the additional parameter for the tf.train.
    This variable is added to LocalParam._auxil_var
    """
    count = 0
    for p in self.sorted_params:
        # Append method if necessary
        if isinstance(p, param.Parameterized) and not hasattr(p, 'set_local_train_var'):
            jself.set_local_train_var(p)
        # Start to set
        if isinstance(p, LocalParam): # Call set_state for local parameter
            count += p.set_local_train_var(x[count:], name)
        elif isinstance(p, param.Param): # Do nothing for global parameter
            pass
        else: # for parameterized
            count += p.set_local_train_var(x[count:], name)
    return count

def get_local_train_var(self, name):
    """
    Recurse get_local_train_var on all child parameters, and hstack them.
    Return: Stacked np-array for all LocalParam
    """
    # check if the child has 'get_local_free_state' method
    for p in self.sorted_params:
        if isinstance(p, (param.Param,param.Parameterized)) and \
        not hasattr(p, 'get_local_train_var'):
            self.set_local_methods(p)
    # Here, additional empty array allows hstacking of empty list
    return np.hstack([p.get_local_train_var(name) for p in self.sorted_params
                       if isinstance(p, (param.Parameterized,LocalParam))]
                     + [np.empty(0, np_float_type)])

class LocalParam(param.Param):
    """
    This is a wrapper class that behaves as free parameters both in the local
    and global optimizers.
    """
    def __init__(self, array, transform=transforms.Identity()):
        param.Param.__init__(self, array, transform)
        # dictionary that stores the  auxiliary variables used in tf.train
        self._train_var = {}

    def get_local_free_state(self):
        return param.Param.get_free_state(self)

    def get_global_free_state(self):
        return []

    def set_local_train_var(self, x, name):
        """
        Given a vector x representing the 'auxiliary' variables of this _array,
        store the result in self._train_var.
        Returns the size of the free parameter.
         """
        if self.fixed:
            self._train_var[name] = np.empty((0,), np_float_type)
            return 0
        free_size = self.transform.free_state_size(self.shape)
        self._train_var[name] = x[:free_size].reshape(self.shape)
        return free_size

    def get_local_train_var(self, name):
        """
        Take the current train_var of this variable.
        This is a numpy method.
        """
        if self.fixed:
            return np.empty((0,), np_float_type)
        return self._train_var[name].flatten()


class GlobalParam(param.Param):
    """
    Simply wrap GPflow.param.Param
    """
    pass

'''
class LocalDataHolder(param.DataHolder):
    """
    Simply wrap GPflow.param.DataHolder so that LocalDataManager can distinguish
    it from global DataHolder
    """
    pass
'''

class HierarchicParameterized(param.Parameterized):
    """
    This class distinguish between local and global parameters in the child
    class.

    This class adds several methods to the child class to provides local and
    global optimizers.
    """
    def __init__(self):
        param.Parameterized.__init__(self)
        # Append methods.
        set_local_methods(self, self)


class LocalDataManager(param.Parentable):
    """
    This object pocesses local data and parameters as np.array and provides some
    of them to model as DataHolder or LocalParam automatically.

    If this object is instanciated in HierarchicParameterized object, it
    automatically search LocalParam in the object.

    self._pickup_data() randomly select the data to be used in the calculation.

    self._store_local_param() methods stores these LocalParam for the current
    data.

    self._restore_local_param() methods restores the previously estimated
    LocalParam for the current data. If the previous result is not available,
    the present result will be used again.

    self.next_set() calls self._pickup_data(), self._store_local_param(),
    self._restore_local_param(), sequencially.

    The typical usage is
    >>> m = GPdict.Param.HierarchicParameterized()
    >>> m.manager = GPdict.param.LocalManager(list_of_dict, minibatch_size=10)
    >>> m.X = m.manager['x'] # set DataHolder to model.
    >>> [... do local optimization ...]
    >>> m.manager.next_set()
    >>> [... do another local optimization ...]

    """
    def __init__(self, list_of_dict, minibatch_size=None, random_seed=1):
        """
        - list_of_dict: list of dictionary that contains the multiple set of data.
                e.g. [{'x':**}, {'y':**}], ...
        - minibatch_size: number of data set that is analyzed once.
        - random_seed: seed for the random number generation that is used to
                pick up data to be analyzed.
        """
        param.Parentable.__init__(self)
        self.data_set = list_of_dict
        self.minibatch_size = minibatch_size
        if minibatch_size is None:
            self.minibatch_size = len(self.data_set)
        # keep data that are currently provided to model as another list.
        self.rng = np.random.RandomState(random_seed)
        self.current_set = self._pickup_data()
        # DataHolder
        self.data_holders = {}
        for key in list_of_dict[0]:
            array = np.stack([d[key] for d in self.current_set])
            self.data_holders[key] = DataHolder(array, 'pass')

    def _pickup_data(self):
        """
        Pick up self.minibatch_size of data from self.data_set
        """
        indices = self.rng.randint(low=0, high=len(self.data_set),
                                                size=self.minibatch_size)
        return [self.data_set[i] for i in indices]

    def _store_local_param(self):
        """
        Store local parameter values.
        They are stored as dict, with key
        - 'value' : stores the array value
        - other key : stores the train_vars
        """
        for p in self.highest_parent.get_local_params():
            for i in range(self.minibatch_size):
                self.current_set[i][p.long_name] = {'value' : p.value[i].copy()}
            for key, item in p._train_var.items():
                for i in range(self.minibatch_size):
                    self.current_set[i][p.long_name][key] = item[i].copy()

    def _restore_local_param(self):
        """
        Restore LocalParam values from previously calculated values (if available)
        """
        for p in self.highest_parent.get_local_params():
            for i in range(self.minibatch_size):
                d = self.current_set[i]
                if p.long_name in d.keys():
                    p._array[i] = d[p.long_name]['value']
            # copy train vars. They sometimes raise KeyError
            for key, item in p._train_var.items():
                if p.long_name in d.keys():
                    p._train_var[key][i] = d[p.long_name][key]

    def next_set(self):
        """
        Store the current LocalParam.value, shuffle the data set, and restore
        the previously calculated LocalParam.value
        """
        self._store_local_param()
        self.current_set = self._pickup_data()
        self._restore_local_param()
        # update data
        for key, item in self.data_holders.items():
            array = np.stack([d[key] for d in self.current_set])
            item.set_data(array)

    def __getitem__(self, key):
        return self.data_holders[key]
