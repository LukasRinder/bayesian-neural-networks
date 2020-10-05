import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from normalizingflows.flow_catalog import Made
from normalizingflows.nf_utils import NormalReparamMNF
from normalizingflows.normalizing_flow import NormalizingFlowModel, NormalizingFlow


tfd = tfp.distributions
tfb = tfp.bijectors


class DenseMNF(tf.keras.layers.Layer):
    """Bayesian fully-connected layer. The weight posterior distribution is modelled by a fully-factorized
    Gaussian. The Gaussian means depend on an auxiliary random variable z, which is modelled by a normalizing flow.
    This allows for multimodality and nonlinear dependencies between the elements of the weight matrix and improves
    significantly upon classical mean field approximation. The flow's base distribution is a normal distribution with
    zero mean and unit variance.

    "Multiplicative Normalizing Flows for Variational Bayesian Neural Networks",
    Christos Louizos, Max Welling (Jun 2017)
    https://arxiv.org/abs/1703.01961
    """

    def __init__(
        self,
        n_out,  # output dimensions
        n_flows_q=2,  # length flow q(z)
        n_flows_r=2,  # length flow r(z|w)
        use_z=True,  # use auxiliary random variable z
        prior_var_w=1,  # variance of weight prior
        prior_var_b=1,  # variance of bias prior
        flow_h_sizes=[32],  # hidden size of flow
        max_std=1.0,  # limit the standard deviation in the forward pass to avoid local minima (e.g. see Louizos et al.)
        **kwargs,
    ):
        self.n_out = n_out
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b
        self.max_std = max_std
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.use_z = use_z
        self.flow_h_sizes = flow_h_sizes
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_in = self.n_in = input_shape[-1]
        # initialization according to He et al. (2015)
        # log variance initialized with N(-9, 0.001) -> e^-9 = 1e-4
        glorot = tf.keras.initializers.GlorotNormal()  # Xavier normal initializer
        mean_init, var_init = -3.0, 1e-3  # -9.0, 1e-3

        # q(w|z): weights and bias separately
        self.mean_W = tf.Variable(glorot([n_in, self.n_out]))
        self.log_var_W = tf.Variable(glorot([n_in, self.n_out]) * var_init + mean_init)

        self.mean_b = tf.Variable(tf.zeros(self.n_out))
        self.log_var_b = tf.Variable(glorot([self.n_out]) * var_init + mean_init)

        if self.use_z:
            # q(z_o): q0_mean has similar function to a dropout rate as it determines the
            # mean of the multiplicative noise z_i in eq. (4)
            self.qz_base = NormalReparamMNF([n_in], var_init=var_init, mean_init=mean_init)

            if n_in > 1:
                permutation = tf.cast(np.concatenate((np.arange(n_in / 2, n_in), np.arange(0, n_in / 2))), tf.int32)

            bijectors_q = []
            for _ in range(self.n_flows_q):
                bijectors_q.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=Made(params=2, hidden_units=self.flow_h_sizes, activation="relu"))))
                if n_in > 1:
                    bijectors_q.append(tfp.bijectors.Permute(permutation))

            self.qz = NormalizingFlowModel(base=self.qz_base, flows=bijectors_q, chain=True, name="qz")

            # r(z|w): c, b1, b2 to compute the mean and std
            self.r0_c = tf.Variable(glorot([n_in]))
            self.r0_b1 = tf.Variable(glorot([n_in]))
            self.r0_b2 = tf.Variable(glorot([n_in]))

            bijectors_r = []
            for _ in range(self.n_flows_r):
                bijectors_r.append(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=Made(params=2, hidden_units=self.flow_h_sizes, activation="relu")))
                if n_in > 1:
                    bijectors_r.append(tfp.bijectors.Permute(permutation))

            self.flow_r = NormalizingFlow(flows=bijectors_r, chain=True)

        self.epsilon_w = tf.Variable(tf.random.normal([self.n_out]), trainable=False)
        self.reset_noise()

    def reset_noise(self):
        # sample new epsilon values
        self.epsilon_w.assign(tf.random.normal([self.n_out]))  # sample epsilon_w
        if self.use_z:
            self.qz.base.reset_noise()  # sample epsilon_z

    def sample_z(self, batch_size, same_noise=False, training=True):
        if self.use_z:
            if training:
                z_samples, log_prob = self.qz.sample(batch_size, same_noise=same_noise)
            else:  # evaluation without noise
                z_samples, log_prob = self.qz.sample_no_noise(batch_size)

        else:
            z_samples = tf.ones([batch_size, self.n_in])
            log_prob = tf.zeros(batch_size)

        return z_samples, log_prob

    @tf.function
    def kl_div(self, same_noise=False):
        z, log_q = self.sample_z(1, same_noise=same_noise)
        log_q = tf.reduce_sum(log_q)

        weight_mu = tf.reshape(z, shape=(self.n_in, 1)) * self.mean_W

        kldiv_weight = 0.5 * tf.reduce_sum((- self.log_var_W + tf.math.exp(self.log_var_W)
                                            + tf.square(weight_mu) - 1))
        kldiv_bias = 0.5 * tf.reduce_sum((- self.log_var_b + tf.math.exp(self.log_var_b)
                                          + tf.square(self.mean_b) - 1))

        log_r = 0
        if self.use_z:
            cw_mu = tf.linalg.matvec(tf.transpose(weight_mu), self.r0_c)
            if same_noise:
                epsilon_w = self.epsilon_w
            else:
                epsilon_w = tf.random.normal([self.n_out])

            cw_var = tf.linalg.matvec(tf.transpose(tf.math.exp(self.log_var_W)), tf.square(self.r0_c))
            cw = tf.math.tanh(cw_mu + tf.math.sqrt(cw_var) * epsilon_w)  # sample W

            mu_tilde = tf.reduce_mean(tf.tensordot(cw, self.r0_b1, axes=0), axis=0)
            neg_log_var_tilde = tf.reduce_mean(tf.tensordot(cw, self.r0_b2, axes=0), axis=0)

            z0, log_r = self.flow_r.inverse(z)
            log_r = tf.reduce_sum(log_r)

            dims = float(z0.shape[-1])
            exponent = tf.squeeze(tf.reduce_sum(tf.square(z0 - mu_tilde) * tf.math.exp(neg_log_var_tilde), axis=1))
            neg_log_det_var = tf.reduce_sum(neg_log_var_tilde)
            log_r += 0.5 * (-dims * tf.math.log(2 * np.pi) + neg_log_det_var - exponent)

        kldiv = kldiv_weight + kldiv_bias + log_q - log_r

        return kldiv

    @tf.function
    def call(self, x, same_noise=False, training=True):
        batch_size = tf.shape(x)[0]
        if training:
            z, _ = self.sample_z(batch_size, same_noise=same_noise)
            mu_out = tf.matmul(x * z, self.mean_W) + self.mean_b

            var_W = tf.clip_by_value(tf.exp(self.log_var_W), 0, self.max_std ** 2)
            var_b = tf.clip_by_value(tf.exp(self.log_var_b), 0, self.max_std ** 2)
            # var_W = tf.square(std_W)
            V_h = tf.matmul(tf.square(x), var_W) + var_b

            if same_noise:  # use the same epsilon per batch
                epsilon_w = tf.expand_dims(self.epsilon_w, axis=0)  # expand batch dimension
                epsilon_w = tf.repeat(epsilon_w, batch_size, axis=0)  # repeat batch dimension
            else:
                epsilon_w = tf.random.normal(tf.shape(mu_out))  # TODO: test implementation

            sigma_out = tf.sqrt(V_h) * epsilon_w

            out = mu_out + sigma_out
        else:  # evaluation without noise
            z, _ = self.sample_z(batch_size, training=training)
            mu_out = tf.matmul(x * z, self.mean_W) + self.mean_b
            out = mu_out

        return out
