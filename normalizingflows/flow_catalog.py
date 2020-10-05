"""
Implementation of various Normalizing Flows.
Tensorflow Bijectors are used as base class. To perform density estimation and sampling, four functions have to be defined
for each Normalizing Flow.


1. _forward:
Turns one random outcome into another random outcome from a different distribution.

2. _inverse:
Useful for 'reversing' a transformation to compute one probability in terms of another.

3. _forward_log_det_jacobian:
The log of the absolute value of the determinant of the matrix of all first-order partial derivatives of the function.

4. _inverse_log_det_jacobian:
The log of the absolute value of the determinant of the matrix of all first-order partial derivatives of the inverse function.


"forward" and "forward_log_det_jacobian" have to be defined to perform sampling.
"inverse" and "inverse_log_det_jacobian" have to be defined to perform density estimation.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

tf.keras.backend.set_floatx('float32')

print('tensorflow: ', tf.__version__)
print('tensorflow-probability: ', tfp.__version__)


'''--------------------------------------- Masked Autoregressive Flow -----------------------------------------------'''


class Made(tfk.layers.Layer):
    """
    Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
    The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
    and one log_scale vector.

    :param params: Python integer specifying the number of parameters to output per input.
    :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
    :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
    :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
    :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
    """

    def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made"):

        super(Made, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
                                                 activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
                                                 bias_regularizer=bias_regularizer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)


'''------------------------------------- Batch Normalization Bijector -----------------------------------------------'''


class BatchNorm(tfb.Bijector):
    """
    Implementation of a Batch Normalization layer for use in normalizing flows according to [Papamakarios et al. (2017)].
    The moving average of the layer statistics is adapted from [Dinh et al. (2016)].

    :param eps: Hyperparameter that ensures numerical stability, if any of the elements of v is near zero.
    :param decay: Weight for the update of the moving average, e.g. avg = (1-decay)*avg + decay*new_value.
    """

    def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            validate_args=validate_args,
            name=name)

        self._vars_created = False
        self.eps = eps
        self.decay = decay

    def _create_vars(self, x):
        # account for 1xd and dx1 vectors
        if len(x.get_shape()) == 1:
            n = x.get_shape().as_list()[0]
        if len(x.get_shape()) == 2: 
            n = x.get_shape().as_list()[1]

        self.beta = tf.compat.v1.get_variable('beta', [1, n], dtype=tf.float32)
        self.gamma = tf.compat.v1.get_variable('gamma', [1, n], dtype=tf.float32)
        self.train_m = tf.compat.v1.get_variable(
            'mean', [1, n], dtype=tf.float32, trainable=False)
        self.train_v = tf.compat.v1.get_variable(
            'var', [1, n], dtype=tf.float32, trainable=False)

        self._vars_created = True

    def _forward(self, u):
        if not self._vars_created:
            self._create_vars(u)
        return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _inverse(self, x):
        # Eq. 22 of [Papamakarios et al. (2017)]. Called during training of a normalizing flow.
        if not self._vars_created:
            self._create_vars(x)

        # statistics of current minibatch
        m, v = tf.nn.moments(x, axes=[0], keepdims=True)
        
        # update train statistics via exponential moving average
        self.train_v.assign_sub(self.decay * (self.train_v - v))
        self.train_m.assign_sub(self.decay * (self.train_m - m))

        # normalize using current minibatch statistics, followed by BN scale and shift
        return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta

    def _inverse_log_det_jacobian(self, x):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(x)
            
        _, v = tf.nn.moments(x, axes=[0], keepdims=True)
        abs_log_det_J_inv = tf.reduce_sum(
            self.gamma - .5 * tf.math.log(v + self.eps))
        return abs_log_det_J_inv
