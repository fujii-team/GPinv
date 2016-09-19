import tensorflow as tf

class LinkFunction:
    def forward(self, F):  # pragma: no cover
        raise NotImplementedError

    def backward(self, F): # pragma: no cover
        raise NotImplementedError

class Identity(LinkFunction):
    def forward(self, F):
        return F

    def backward(self, F):
        return F

class Log(LinkFunction):
    def __init__(self, lower=1e-30):
        """
        - lower is a float variable that improve the stability where F
        approaches significantly to zero. Default value is 1.0e-30
        """
        self.lower = lower

    def forward(self, F):
        return tf.log(F+self.lower)

    def backward(self, F):
        return tf.exp(F)
