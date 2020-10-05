import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class BayesByBackprop(tf.keras.layers.Layer):
    """Bayesian fully-connected layer. The weight posterior distribution is modelled by a fully-factorized
    Gaussian.

    "Weight Uncertainty in Neural Networks" - Blundell et al. (2015)
    https://arxiv.org/abs/1505.05424
    """

    def __init__(
        self,
        n_out,  # output dimensions
        prior_var_w=1,  # variance of weight prior
        prior_var_b=1,  # variance of bias prior
        max_std=1.0,  # limit the standard deviation in the forward pass to avoid local minima (e.g. see Louizos et al.)
        log_var_mean_init=-3.0,
        log_var_init=1e-3,
        **kwargs,
    ):
        self.n_out = n_out
        self.prior_var_w = prior_var_w
        self.prior_var_b = prior_var_b
        self.max_std = max_std
        self.log_var_mean_init = log_var_mean_init
        self.log_var_init = log_var_init
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_in = self.n_in = input_shape[-1]
        # initialization according to He et al. (2015)
        # log variance initialized with N(-9, 0.001) -> e^-9 = 1e-4
        glorot = tf.keras.initializers.GlorotNormal()  # Xavier normal initializer
        mean_init, var_init = self.log_var_mean_init, self.log_var_init  # -9.0, 1e-3

        self.mean_W = tf.Variable(glorot([n_in, self.n_out]))
        self.log_var_W = tf.Variable(glorot([n_in, self.n_out]) * var_init + mean_init)

        self.mean_b = tf.Variable(tf.zeros(self.n_out))
        self.log_var_b = tf.Variable(glorot([self.n_out]) * var_init + mean_init)

        self.epsilon_w = tf.Variable(tf.random.normal([self.n_out]), trainable=False)
        self.reset_noise()

    def reset_noise(self):
        # sample new epsilon values
        self.epsilon_w.assign(tf.random.normal([self.n_out]))  # sample epsilon_w

    @tf.function
    def kl_div(self, same_noise=True):
        kldiv_weight = 0.5 * tf.reduce_sum((- self.log_var_W + tf.math.exp(self.log_var_W)
                                            + tf.square(self.mean_W) - 1))
        kldiv_bias = 0.5 * tf.reduce_sum((- self.log_var_b + tf.math.exp(self.log_var_b)
                                          + tf.square(self.mean_b) - 1))

        kldiv = kldiv_weight + kldiv_bias

        return kldiv

    @tf.function
    def call(self, x, same_noise=False, training=True):
        batch_size = tf.shape(x)[0]
        if training:
            mu_out = tf.matmul(x, self.mean_W) + self.mean_b

            var_W = tf.clip_by_value(tf.exp(self.log_var_W), 0, self.max_std ** 2)
            var_b = tf.clip_by_value(tf.exp(self.log_var_b), 0, self.max_std ** 2)

            V_h = tf.matmul(tf.square(x), var_W) + var_b

            if same_noise:  # use the same epsilon per batch
                epsilon_w = tf.expand_dims(self.epsilon_w, axis=0)  # expand batch dimension
                epsilon_w = tf.repeat(epsilon_w, batch_size, axis=0)  # repeat batch dimension
            else:
                epsilon_w = tf.random.normal(tf.shape(mu_out))

            sigma_out = tf.sqrt(V_h) * epsilon_w

            out = mu_out + sigma_out
        else:  # evaluation without noise
            mu_out = tf.matmul(x, self.mean_W) + self.mean_b
            out = mu_out

        return out
