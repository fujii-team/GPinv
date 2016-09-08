import tensorflow as tf
import numpy as np
from GPflow.param import Param, DataHolder
from GPflow.model import GPModel
from GPflow import transforms
from GPflow.mean_functions import Zero
from GPflow.tf_hacks import eye
from param import DiagL, MultiDiagL

class VGP(GPModel):
    """
    This is an extended version of the Variational Gaussian Process(VGP).
    The key reference is:
    @article{Opper:2009,
        title = {The Variational Gaussian Approximation Revisited},
        author = {Opper, Manfred and Archambeau, Cedric},
        journal = {Neural Comput.},
        year = {2009},
        pages = {786--792},
    }

    The main difference is the adoption of the likelihood which have
    correlation among latent functions.

    Due to this correlation, the diagonal part of the non-diagonal-elements
    also becomes parameters,

    q(f) = N(f | K alpha, [K^-1 + L]^-1)

    where f = [f0, f1, ..., fM, g0, g1, ..., gL]

    L is a matrix that may have off-diagonal part.
    """
    def __init__(self, X, Y,
                 kern, likelihood, mean_function=Zero(),
                 mode='mean_field',
                 semidiag_list=None,
                 num_latent=None):
        """
        X: data that are passed to kernel and mean_function
        Y: data that are passed to likelihood.
        kern, likelihood, mean_function are appropriate GPflow objects

        mode : one of [mean_field, full_rank, semi_diag]
        if mode == 'semi_diag', offdiag_indices should be specified.

        semidiag_objects: list of dict.
            semidiag_list[i]['head_index'] : 2-integer tuple that specifies
                                                the left-top index for each semi-diag part.
            semidiag_list[i]['length'] : length of each semi-diag part.
        """
        # assert that X and Y have the same length
        assert mode in ['mean_field', 'full_rank', 'semi_diag']
        if mode == 'semi_diag' and semidiag_list is None:
            raise('semidiag_list should be provided in semi_diag mode')

        # data
        X = DataHolder(X, on_shape_change='raise')
        Y = DataHolder(Y, on_shape_change='raise')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)

        # parameters
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        diag_indices = np.array([[i,i] for i in range(self.num_data)])
        if mode is 'mean_field':
            self.q_lambda = DiagL(np.ones((self.num_data, self.num_latent)))
        else:
            if mode is 'full_rank':
                # TODO add the full-rank support
                raise NotImplementedError
            else:
                #TODO **IMPORTANT** offdiag_correction should be added.
                # i.e, if both [i, j] and [i, j'] are correlated (i' < i),
                # then [i,i'] is also non-zero in L,
                head_indices = [[0,0],]+[obj['head_index'] for obj in semidiag_list]
                values = [np.ones((self.num_data, self.num_latent))]+\
                         [np.zeros((obj['length'],self.num_latent)) for obj in semidiag_list]
                trans = [transforms.positive]+[transforms.Identity()]*len(semidiag_list)

            self.q_lambda = MultiDiagL(head_indices, values, trans,
                        shape=[self.num_data, self.num_data, self.num_latent])

        self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))


    def build_likelihood(self):
        """
        q_alpha, q_lambda are variational parameters, size N x R
        This method computes the variational lower bound on the likelihood,
        which is:
            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]
        with
            q(f) = N(f | K alpha + mean, [K^-1 + diag(square(lambda))]^-1) .
        """
        K = self.kern.K(self.X)
        K_alpha = tf.matmul(K, self.q_alpha) # shape [N, num_latent]
        f_mean = K_alpha + self.mean_function(self.X)
        # Cholesky factor of the correaltion part of the covariance.
        Llam = self.q_lambda.L() # shape [num_latent, num_data, num_data]
        Llam_inv = self.q_lambda.Linv()

        # shape [num_latent, num_data, num_data]
        I = tf.tile(tf.expand_dims(eye(self.num_data), 0), [self.num_latent, 1, 1])
        # shape [num_latent, num_data, num_data]
        K = tf.tile(tf.expand_dims(K, 0), [self.num_latent, 1, 1])
        # A = I + Llam K Llam^T
        A = I + tf.batch_matmul(tf.batch_matmul(Llam, K), Llam, adj_y=True)
        L = tf.batch_cholesky(A)
        Li = tf.batch_matrix_triangular_solve(L, I, lower=True)

        A_logdet = 2*tf.reduce_sum(tf.log(tf.batch_matrix_diag_part(L)))
        trAi = tf.reduce_sum(tf.square(Li))

        KL = 0.5 * (A_logdet + trAi - self.num_data
                    + tf.reduce_sum(K_alpha*self.q_alpha))

        # Posterior covariance
        # TODO Rank-two update should be applied to evaluate Lcov.
        Ai = tf.batch_matmul(Li, Li, adj_y=True) # A^-1
        # Here, 1.0e-9 * eye is added for stability.
        Lcov_tmp = tf.batch_cholesky(I - Ai + I*1.0e-12) # (I - A^-1)^(1/2)
        Lcov = tf.batch_matmul(Llam_inv, Lcov_tmp)

        # Lcov.shape = [num_data, num_data, num_latent]
        v_exp = self.likelihood.stochastic_expectations(
                f_mean, tf.transpose(Lcov, [1,2,0]), self.Y)
        return tf.reduce_sum(v_exp) - KL


    def build_predict(self, Xnew, full_cov=False):
        """
        The posterior variance of F is given by
            q(f) = N(f | K alpha + mean, [K^-1 + diag(lambda**2)]^-1)
        Here we project this to F*, the values of the GP at Xnew which is given
        by
           q(F*) = N ( F* | K_{*F} alpha + mean, K_{**} - K_{*f}[K_{ff} +
                                           diag(lambda**-2)]^-1 K_{f*} )
        """
        # compute kernel things
        Kx = self.kern.K(self.X, Xnew) # shape=[N,N']
        K = self.kern.K(self.X)        # shape=[N,N]

        # Cholesky factor of the correaltion part of the covariance.
        Llam_inv = self.q_lambda.Linv()
        # predictive mean
        f_mean = tf.matmul(tf.transpose(Kx), self.q_alpha) + self.mean_function(Xnew)

        # shape [num_latent, num_data, num_data]
        I = tf.tile(tf.expand_dims(eye(self.num_data), 0), [self.num_latent, 1, 1])
        K = tf.tile(tf.expand_dims(K, 0), [self.num_latent, 1, 1])
        # A = I + Llam^T K Llam
        A = K + tf.batch_matmul(Llam_inv, Llam_inv, adj_y=True)
        L = tf.batch_cholesky(A)
        # shape [num_latent, num_data, N']
        Kx_tiled = tf.tile(tf.expand_dims(Kx, 0), [self.num_latent, 1, 1])
        # shape [num_latent, num_data, N']
        LiKx = tf.batch_matrix_triangular_solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kern.K(Xnew) - tf.batch_matmul(LiKx, LiKx, adj_x=True)
        else:
            f_var = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(LiKx), 1)
        return f_mean, tf.transpose(f_var)
