import tensorflow as tf
import numpy as np
import GPinv
import GPflow

def make_LosMatrix(r,z):
    """
    Constructing a matrix for a cylindrical plasma coordinate.
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


class AbelLikelihood(GPinv.likelihoods.NonLinearLikelihood):
    def __init__(self, Amat, num_stocastic_points=20):
        GPinv.likelihoods.NonLinearLikelihood.__init__(self, num_stocastic_points)

        self.Amat = GPflow.param.DataHolder(Amat)
        self.variance = GPflow.param.Param(np.ones(1), GPflow.transforms.positive)

    def log_prob(self, Xlist, Ylist):
        """
        The log_probability for this Abel's likelihood.
        This part should be implemented in the child class.
        :param list of tensor Xlist: list of the latent functions with length Q.
                The shape of the i-th element is [Ni,M]
        :param list of tensor Ylist: list of the observations with length P.
                The shape of the i-th element is [Ni',M']
        :return list of log of the likelihood with length P.
            The shape should be the same to that of Ylist.
        """
        F = tf.matmul(self.Amat, Xlist[0])
        Y = Ylist[0]
        return [GPflow.densities.gaussian(Y, F, self.variance)]
