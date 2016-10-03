import tensorflow as tf
from functools import reduce
from GPflow import mean_functions
from GPflow.param import ParamList

class Zero(mean_functions.Zero):
    """ Just a wrap """
    pass

class Constant(mean_functions.Constant):
    """ Just a wrap """
    pass

class Linear(mean_functions.Linear):
    """ Just a wrap """
    pass
'''
class SwitchedMeanFunction(mean_functions.MeanFunction):
    """
    This class enables to use different (independent) mean_functions respective
    to the data group.
    ! NOTE !
    This MeanFunction accepts param.ConcatParamList or param.ConcatDataHolder as
    arguments, rather than tf.tensor.
    """
    def __init__(self, meanfunction_list):
        mean_functions.MeanFunction.__init__(self)
        for m in meanfunction_list:
            assert isinstance(m, mean_functions.MeanFunction)
        self.meanfunction_list = ParamList(meanfunction_list)

    def __call__(self, X):
         return reduce(tf.add,
            [tf.pad(mean(x), [[begin, X.shape[0]-begin-tf.shape(x)[0]],[0,0]])
                for mean,x,begin
                in zip(self.meanfunction_list, X, X.slice_begin)])
'''
