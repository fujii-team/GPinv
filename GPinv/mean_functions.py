import tensorflow as tf
from GPflow import mean_functions
from GPflow.param import ParamList

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
    def __init__(self, meanfunction_list):
        mean_functions.MeanFunction.__init__(self)
        for m in meanfunction_list:
            assert isinstance(m, mean_functions.MeanFunction)
        self.meanfunction_list = ParamList(meanfunction_list)
        self.num_meanfunctions = len(meanfunction_list)

    def __call__(self, X):
        ind = tf.gather(tf.transpose(X), tf.shape(X)[1]-1)  # ind = X[:,-1]
        ind = tf.cast(ind, tf.int32)
        X = tf.transpose(tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1]-1)))  # X = X[:,:-1]

        # split up X into chunks corresponding to the relevant likelihoods
        x_list = tf.dynamic_partition(X, ind, self.num_meanfunctions)
        # apply the likelihood-function to each section of the data
        results = [m(x) for (x,m) in zip(x_list, self.meanfunction_list)]
        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_meanfunctions)
        return tf.dynamic_stitch(partitions, results)
