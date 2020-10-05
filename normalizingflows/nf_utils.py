"""
Implementation of functions that are important for training normalizing flows.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


'''----------------------------------- Normal distribution with reparametrization -----------------------------------'''


class NormalReparamMNF(tf.Module):
    """
    Normal distribution with reparameterization to be able to learn the mean and variance.

    :param shape: Shape of the tensor
    :param std_init (float): initialization value for the standard deviation, optional
    :param mean_init (float): initialization value for the mean, optional
    """
    def __init__(self, shape, var_init=1.0, mean_init=0.0):
        super(NormalReparamMNF, self).__init__()

        glorot = tf.keras.initializers.GlorotNormal()  # Xavier normal initializer

        self.shape = shape
        self.mean = tf.Variable(glorot(shape), trainable=True)
        self.log_var = tf.Variable(glorot(shape) * var_init + mean_init, trainable=True)
        self.epsilon = tf.Variable(tf.random.normal(self.shape), trainable=False)

    @tf.function
    def sample(self, batch_size, same_noise=False):
        mean = tf.tile(self.mean[None, :], [batch_size, 1])  # split tensor into batches
        if same_noise:
            epsilon = tf.expand_dims(self.epsilon, axis=0)  # expand batch size dimension
            epsilon = tf.repeat(epsilon, batch_size, axis=0)  # use the same noise/epsilon for the whole batch
        else:
            epsilon = tf.random.normal([batch_size, self.shape[0]])
        var = tf.exp(self.log_var)
        samples = mean + tf.sqrt(var) * epsilon

        return samples

    @tf.function
    def log_prob(self, samples):
        dims = float(samples.shape[-1])
        var = tf.exp(self.log_var)
        exponent = tf.reduce_sum(tf.square(samples - self.mean)/var, axis=1)
        log_det_var = tf.reduce_sum(self.log_var)
        log_prob = -0.5 * (dims * tf.math.log(2 * np.pi) + log_det_var + exponent)

        return log_prob

    def prob(self, samples):
        log_prob = self.log_prob(samples)

        return tf.exp(log_prob)

    def log_std(self):
        return 0.5 * self.log_var

    def reset_noise(self):
        self.epsilon.assign(tf.random.normal(self.shape))
