import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer

tfkl = tf.keras.layers


class Backbone(tf.keras.Model):
    """
    Backbone of the Deep Q-Network (DQN) with Bayesian fully-connected layers that approximates the Q-function.
    The Bayesian fully-connected layers utilize Dropout as Bayesian approximation according to
    "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
    - Gal and Ghahramani (2015): https://arxiv.org/abs/1506.02142.

    Takes 'num_states' inputs and outputs one Q-value for each action.
    """
    def __init__(self, num_states, hidden_units, dropout_rate, num_actions, N):
        super(Backbone, self).__init__()

        self.N = N  # data points
        lengthscale = 1e-2
        tau = 1.0
        reg = lengthscale**2 * (1 - dropout_rate) / (2.0 * self.N * tau)

        self.hidden_layers = []
        self.input_layer = InputLayer(input_shape=(num_states,))
        for i in hidden_units:
            self.hidden_layers.append(tfkl.Dense(i, activation='relu', kernel_initializer='RandomNormal',
                                                            kernel_regularizer=tf.keras.regularizers.L1L2(l2=reg)))

        self.hidden_layers.append(tfkl.Dropout(dropout_rate))  # only one dropout layer before the output

        self.output_layer = tfkl.Dense(num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        out = self.input_layer(inputs)

        for layer in self.hidden_layers:
            if isinstance(layer, tfkl.Dropout):
                out = layer(out, training=True)
            else:
                out = layer(out)
        out = self.output_layer(out)
        return out


class DQN(tf.Module):
    """
    Deep Q-Network utilizing Dropout as Bayesian approximation for efficient sampling.
    """
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr, dropout_rate):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = Backbone(num_states, hidden_units, dropout_rate, num_actions, max_experiences)
        self.experience = {'s': [], 'a': [], 'r': [], 's_next': [], 'end': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.states_uncertainty = {}

    def predict(self, inputs, training=True):
        """
        Get Q-values from backbone network.
        :param inputs: inputs for the backbone network, e.g. states.
        :param training: forward pass without stochasticity, if set to `False`.
        :return: outputs of the backbone network, e.g. num_action Q-values.
        """
        return self.model(tf.convert_to_tensor(inputs, tf.float32), training=training)

    def train(self, target_net):
        """
        Train with experience replay, e.g. replay using a randomized order removing correlation in observation sequence
        to deal with biased sampling
        :param target_net: target network.
        """
        if len(self.experience['s']) < self.min_experiences:
            return 0, 0

        experience_replay_enabled = True  # set False to disable experience replay
        if experience_replay_enabled:
            # sample random minibatch of transitions
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        else:
            n = len(self.experience['s'])
            if n < self.batch_size:
                ids = np.full(self.batch_size, n-1)
            else:
                ids = np.arange(max(0, n - self.batch_size), (n - 1), 1)

        states = tf.convert_to_tensor([self.experience['s'][i] for i in ids], tf.float32)
        actions = tf.convert_to_tensor([self.experience['a'][i] for i in ids], tf.float32)
        rewards = tf.convert_to_tensor([self.experience['r'][i] for i in ids], tf.float32)
        states_next = tf.convert_to_tensor([self.experience['s_next'][i] for i in ids], tf.float32)
        ends = tf.convert_to_tensor([self.experience['end'][i] for i in ids], tf.bool)

        # compute loss and perform gradient descent
        loss, reg_loss = self.gradient_update(target_net, states, actions, rewards, states_next, ends)

        return loss, reg_loss

    @tf.function
    def gradient_update(self, target_net, states, actions, rewards, states_next, ends):
        """
        Gradient update with @tf.function decorator for faster performance.
        """
        # make predictions with target network and get sample q for Q-function update, sample is different if epoch end
        double_dqn = True
        if double_dqn:
            next_action = tf.math.argmax(self.predict(states_next), axis=1)
            q_values = target_net.predict(states_next)
            q_max = tf.math.reduce_sum(q_values * tf.one_hot(next_action, self.num_actions), axis=1)
        else:
            q_max = tf.math.reduce_max(target_net.predict(states_next), axis=1)

        y = tf.where(ends, rewards, rewards + self.gamma * q_max)

        # perform gradient descent
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)

            # Q-values from training network for selected actions
            q_values = self.predict(states)
            selected_q_values = tf.math.reduce_sum(q_values * tf.one_hot(tf.cast(actions, tf.int32), self.num_actions), axis=1)

            regularization_loss = tf.reduce_sum(self.model.losses)
            loss_pred = tf.math.reduce_sum(tf.square(y - selected_q_values))  # compute loss
            loss = loss_pred + regularization_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, regularization_loss

    def get_action(self, states, training=True):
        """
        Predict action with the MC Dropout network. Keeping MC Dropout enabled in the forward pass forms a Bayesian
        approximation. Hence, approximated Thompson sampling is performed.

        :param states: observed states, e.g. [x, dx, th, dth].
        :param training: forward pass without stochasticity, if set to `False`.
        :return: action
        """
        q_values = self.predict(np.atleast_2d(states), training)
        action = np.argmax(q_values)
        return action

    def add_experience(self, exp):
        """
        Add experience to experience history. If 'max_experiences' exceeded, remove first item and append current
        experience.
        :param exp: experience {'s': prev_observations, 'a': action, 'r': reward, 's_next': observations, 'end': end}.
        """
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)

        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, train_net):
        """
        Copy weights from train network to target network.
        :param train_net: model of train network.
        """
        variables_target = self.model.trainable_variables
        variables_train = train_net.model.trainable_variables

        for v_target, v_train in zip(variables_target, variables_train):
            v_target.assign(v_train.numpy())
