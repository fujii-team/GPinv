import tensorflow as tf
import numpy as np
import GPinv
import GPflow

def make_LosMatrix(r,z):
    """
    Construct a matrix for a cylindrical plasma coordinate.
    A[i,j] element stores a passing length for a shell j by chord i.

    r: radius coordinate
    z: los position.
    """
    n = r.shape[0]
    N = z.shape[0]
    dr = r[1] - r[0]
    A = np.zeros((N,n))
    for i in range(N):
        for j in range(1, n-1):
            # inner radius of the shell j
            r_in  = 0.5*(r[j-1]+r[j])
            # outer radius of the shell j
            r_out = 0.5*(r[j+1]+r[j])
            # if the code passes the shell j
            if np.abs(z[i]) < r_out:
                # if the chord passes both the inner and outer sides
                if np.abs(z[i]) < r_in:
                    A[i,j] = 2.*np.sqrt(r_out**2. - z[i]**2) - \
                             2.*np.sqrt(r_in **2. - z[i]**2)
                # if the chord passes only the outer side
                elif np.abs(z[i]) > r_in:
                    A[i,j] = 2.*np.sqrt(r_out **2. - z[i]**2)

        # inner radius of the shell j
        r_in  = 0.5*(r[n-2]+r[n-1])
        # outer radius of the shell j
        r_out = r[n-1]
        if np.abs(z[i]) < r_out:
            if np.abs(z[i]) < r_in:
                # inner radius of the shell j
                A[i,n-1] = 2.*np.sqrt(r_out**2. - z[i]**2) - \
                         2.*np.sqrt(r_in **2. - z[i]**2)
                # if the chord passes only the outer side
            elif np.abs(z[i]) > r_in:
                A[i,n-1] = 2.*np.sqrt(r_out **2. - z[i]**2)
    return A

def make_cosTheta(r,z):
    """
    Construct a matrix for a cylindrical plasma coordinate.
    cos(theta[i,j]) element stores a passing length for a shell j by chord i.

    r: radius coordinate
    z: los position.
    """
    n = r.shape[0]
    N = z.shape[0]
    cosTheta = np.zeros((N,n))
    for i in range(N):
        for j in range(n):
            if np.abs(z[i]) < r[j]:
                cosTheta[i,j] = z[i]/r[j]
    return cosTheta


import tensorflow as tf

class AbelLikelihood(GPinv.likelihoods.TransformedLikelihood):
    def __init__(self, Amat, num_samples=20):
        GPinv.likelihoods.TransformedLikelihood.__init__(
                        self, num_samples, link_func=GPinv.link_functions.Log())

        self.Amat = GPinv.param.DataHolder(Amat)
        self.variance = GPinv.param.Param(np.ones(1), GPinv.transforms.positive)

    def transform(self, F):
        Amat = tf.tile(tf.expand_dims(self.Amat, [0]), [tf.shape(F)[0], 1,1])
        return tf.batch_matmul(Amat, tf.exp(F))

    def log_p(self, X, Y):
        """
        :param list of tensor Xlist: tensor for the latent function.
                The shape of the i-th element is [Ni,M]
        :param list of tensor Ylist: tensor for the observation.
                The shape of the i-th element is [Ni',M]
        :return list of log of the likelihood with length P.
            The shape should be the same to that of Ylist.
        """
        return GPinv.densities.gaussian(X, Y, self.variance)
