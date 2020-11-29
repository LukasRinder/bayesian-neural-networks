import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from bayes.ConcreteDropout import ConcreteDropout


def make_backbone(num_states, hidden_units, num_actions, dropout_reg=1e-5, wd=1e-3):
    """
    Build a tensorflow keras backbone model utilizing concrete dropout layers.
    """
    losses: list = []
    inp = Input(shape=(num_states,))
    x = inp

    for i in hidden_units:
        x, loss = ConcreteDropout(Dense(i, activation='relu'),
                                  weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
        losses.append(loss)

    x = Dense(100, activation='relu')(x)
    out = Dense(num_actions, activation='linear')(x)
    model = Model(inp, out)
    model.add_loss(losses)

    return model
    

class DQN(tf.Module):
    """
    Deep Q-Network.
    """
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.SGD(lr)
        self.gamma = gamma
        self.model = make_backbone(num_states, hidden_units, num_actions)
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
        Predict action with the Concrete Dropout network. Keeping Concrete Dropout enabled in the forward pass forms a
        Bayesian approximation. Hence, approximated Thompson sampling is performed.

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
