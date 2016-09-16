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

class SwitchedMeanFunction(mean_functions.MeanFunction):
    """
    This class enables to use different (independent) mean_functions respective
    to the data 'label'.
    We assume the 'label' is stored in the extra column of X.
    """
    def __init__(self, meanfunction_list, slice_X_begin, slice_X_size):
        mean_functions.MeanFunction.__init__(self)
        for m in meanfunction_list:
            assert isinstance(m, mean_functions.MeanFunction)
        self.meanfunction_list = ParamList(meanfunction_list)
        # store slice data
        self.slice_X_begin, self.slice_X_size = slice_X_begin, slice_X_size

    def __call__(self, X):
         return reduce(tf.add,
            [tf.pad(mean(tf.slice(X, [begin,0], [size, -1])),
                    [[begin,tf.shape(X)[0]-begin-size],[0,0]])
                for mean,begin,size
                in zip(self.meanfunction_list, self.slice_X_begin, self.slice_X_size)])
