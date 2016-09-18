# This is a modification of GPflow/conditionals.py by Keisuke Fujii.
#
# The original source file is distributed at
# https://github.com/GPflow/GPflow/blob/master/GPflow/conditionals.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from GPflow.tf_wraps import eye
import tensorflow as tf
from GPflow.scoping import NameScoped
from GPflow._settings import settings


@NameScoped("multilatent_conditional")
def conditional(Xnew, X, kern, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    This is the modified version of GPflow/conditionals.py.
    The only difference is
    + Xnew is an instance of param.ConcatDataHolder not tf.tensor
    + X is an instance of param.ConcatDataHolder not tf.tensor
    + tf.cholesky(Kmm) is replaced by kern.cholesky(X)
    """

    # compute kernel stuff
    num_data = X.shape[0] #tf.shape(X)[0]
    Kmn = kern.K(X, Xnew)
    Lm = kern.Cholesky(X)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
        shape = tf.pack([tf.shape(f)[1], 1, 1])
    else:
        fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
        shape = tf.pack([tf.shape(f)[1], 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # D x N x N or D x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(tf.transpose(A), f)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # D x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.batch_matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # D x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.pack([tf.shape(f)[1], 1, 1]))
            LTA = tf.batch_matmul(L, A_tiled, adj_x=True)  # D x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.batch_matmul(LTA, LTA, adj_x=True)  # D x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # D x N
    fvar = tf.transpose(fvar)  # N x D or N x N x D

    return fmean, fvar
