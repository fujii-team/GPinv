import tensorflow as tf
import numpy as np
from GPflow import mean_functions
from GPflow.param import Param,ParamList
from GPflow._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class MeanFunction(mean_functions.MeanFunction):
    """
    A wrap of GPflow.mean_functions.MeanFunction.
    The main difference of this wrap is __call__ method, that returns
    nxR sized tensor, in contrast to GPflow.mean_functions.MeanFunction, which
    returns nx1 sized tensor.
    """
    def __init__(self, output_dim):
        """
        :param integer output_dim: number of output dimension, R.
        """
        mean_functions.MeanFunction.__init__(self)
        self.output_dim = output_dim

    def __call__(self, X):
        """
        :param tf.tensor x: nxD tensor.
        :return tf.tensor: nxR tensor.
        """
        raise NotImplementedError("Implement the __call__\
                                  method for this mean function")

class Zero(MeanFunction):
    """ Zero mean """
    def __call__(self, X):
        return tf.zeros([self.output_dim, tf.shape(X)[0]], float_type)

class Constant(MeanFunction):
    """ Constant mean """
    def __init__(self, output_dim, c=None):
        MeanFunction.__init__(self, output_dim)
        if c is None:
            c = np.ones(output_dim,np_float_type)
        self.c = Param(c)

    def __call__(self, X):
        return tf.tile(tf.expand_dims(self.c,-1), [1,tf.shape(X)[0]])

class Stack(MeanFunction):
    """
    Mean function that returns multiple kinds of mean values, stacked
    vertically.

    Input for the initializer is a list of MeanFunctions, [m_1,m_2,...,m_M].
    The function call returns [m_1(X),m_2(X),...,m_M(X)].
    The size of the return is n x (sum_i m_i.output_dim).
    """
    def __init__(self, list_of_means):
        """
        :param list list_of_means: A list of MeanFunction object.
        """
        output_dim = 0
        for m in list_of_means:
            output_dim += m.output_dim
        MeanFunction.__init__(self, output_dim)
        # MeanFunctions are stored as ParamList
        self.mean_list = ParamList(list_of_means)

    def __call__(self, X):
        """
        Return a concatenated tensor of the multiple mean functions.
        The size of the return is n x (sum_i m_i.output_dim).
        """
        return tf.concat(0, [l(X) for l in self.mean_list])
