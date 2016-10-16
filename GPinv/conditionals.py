import tensorflow as tf
from GPflow.tf_wraps import eye
from GPflow.scoping import NameScoped
from GPflow._settings import settings

@NameScoped("conditional")
def conditional(Xnew, X, kern, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Given F, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.
    Additionally, there my be Gaussian uncertainty about F as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.
    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.
    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).
    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).
     - Xnew is a data matrix, size n x D
     - X are data points, size m x D
     - kern is a GPinv kernel
     - f is a data matrix, m x R, representing the function values at X, for R functions.
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size m x R or m x m x R
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.
    These functions are now considered deprecated, subsumed into this one:
        gp_predict
        gaussian_gp_predict
        gp_predict_whitened
        gaussian_gp_predict_whitened
    """
    # compute kernel stuff
    num_data = tf.shape(X)[0]
    Kmn = kern.K(X, Xnew) # [R,n,n2]
    Lm  = kern.Cholesky(X) # [R,n,n]

    # Compute the projection matrix A
    A = tf.batch_matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov: # shape [R,n,n]
        fvar = kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
    else:        # shape [R,n]
        fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 1)

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.batch_matrix_triangular_solve(Lm, A, lower=False)

    # change shape of f [m,R] -> [R,m,1]
    f = tf.expand_dims(f, -1)
    # construct the conditional mean, sized [m,R]
    fmean = tf.transpose(tf.squeeze(tf.batch_matmul(A, f, adj_x=True), [-1]))

    if q_sqrt is not None:
        # diagonal case.
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x m x n
        # full cov case
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.batch_matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # D x M x M
            LTA = tf.batch_matmul(L, A, adj_x=True)  # R x m x n
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.batch_matmul(LTA, LTA, adj_x=True)  # R x n x n
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x n
    #fvar = tf.transpose(fvar)  # n x R or n x n x R

    return fmean, fvar
