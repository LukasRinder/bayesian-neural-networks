import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors


class NormalizingFlow(tf.Module):
    """
    Stacking of several normalizing flows. Constitutes a normalizing flow itself.
    """

    def __init__(self, flows, chain=True, name=None, **kwargs):
        super(NormalizingFlow).__init__(**kwargs)
        if not isinstance(name, str):
            name = "flow"

        self.flows = flows
        self.chain = chain  # use tfb.Chain
        if chain:
            self.flow = tfb.Chain(bijectors=list(reversed(flows)), name=name)

    @tf.function
    def forward(self, z):  # z -> x
        if self.chain:
            x = self.flow.forward(z)
            log_dets = self.flow.forward_log_det_jacobian(z, event_ndims=1)
        else:
            log_dets = tf.zeros(tf.shape(z)[0])
            zk = z
            for flow in self.flows:
                log_dets = log_dets + flow._forward_log_det_jacobian(zk)  # "-" already in forward_log_det_jacobian
                zk = flow.forward(zk)

            x = zk

        return x, log_dets

    @tf.function
    def inverse(self, x):  # x -> z
        if self.chain:
            z = self.flow.inverse(x)
            log_dets = self.flow.inverse_log_det_jacobian(x, event_ndims=1)
        else:
            log_dets = tf.zeros(tf.shape(x)[0])
            zk = x
            for flow in reversed(self.flows):
                log_dets = log_dets + flow._inverse_log_det_jacobian(zk)
                zk = flow.inverse(zk)

            z = zk

        return z, log_dets


class NormalizingFlowModel(NormalizingFlow):
    """A normalizing flow model as a combination of base distribution and flow."""

    def __init__(self, base, flows, name="transformed_dist", **kwargs):
        super().__init__(flows, name=name, **kwargs)

        self.base = base  # distribution class that exposes a log_prob() and sample() method
        self.flows = flows

    def log_prob(self, x):
        z, log_dets = self.inverse(x)
        base_prob = self.base.log_prob(z)

        return base_prob + log_dets

    def prob(self, x):
        return tf.exp(self.log_prob(x))

    def sample(self, batch_size, same_noise=False):
        z = self.base.sample(batch_size, same_noise=same_noise)
        base_prob = self.base.log_prob(z)
        x, log_dets = self.forward(z)

        return x, base_prob + log_dets

    def sample_no_noise(self, batch_size):
        z = tf.expand_dims(self.base.mean, axis=0)  # expand batch dimension
        z = tf.repeat(z, batch_size, axis=0)
        base_prob = self.base.log_prob(z)
        x, log_dets = self.forward(z)

        return x, base_prob + log_dets
