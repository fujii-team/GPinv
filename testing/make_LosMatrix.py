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
