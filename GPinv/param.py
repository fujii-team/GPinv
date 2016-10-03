import tensorflow as tf
import numpy as np
from GPflow import param
from GPflow import svgp

class MinibatchData(svgp.MinibatchData):
    """
    This is a simple wrap of GPflow.svgp.MinibatchData.
    The additional _alldata flag is prepared for the predicting phase.
    """
    def __init__(self, array, minibatch_size, rng=None):
        svgp.MinibatchData.__init__(self, array, minibatch_size, rng)
        self._alldata = False

    def get_feed_dict(self):
        """
        Return minibatch if self._alldata is not raised.
        Otherwise, it returns all the data.
        """
        if not self._alldata:
            return svgp.MinibatchData.get_feed_dict(self)
        else:
            return {self._tf_array: self._array}


class Param(param.Param):
    pass

class DataHolder(param.DataHolder):
    pass

class Parameterized(param.Parameterized):
    pass

class ParamList(param.ParamList):
    pass
