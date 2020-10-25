import numpy as np
import tensorflow as tf

tfkl = tf.keras.layers


class Backbone(tf.keras.Model):
    """
    Backbone of the Deep Q-Network (DQN) that approximates the Q-function.
    Takes 'num_states' inputs and outputs one Q-value for each action.
    """
    def __init__(self, num_states, hidden_units, num_actions):
        super(Backbone, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN(tf.Module):
    """
    Deep Q-Network.
    """
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = Backbone(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's_next': [], 'end': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        """
        Get Q-values from backbone network.
        :param inputs: inputs for the backbone network, e.g. states.
        :return: outputs of the backbone network, e.g. num_action Q-values.
        """
        return self.model(tf.convert_to_tensor(inputs, tf.float32))

    def train(self, target_net):
        """
        Train with experience replay, e.g. replay using a randomized order removing correlation in observation sequence
        to deal with biased sampling
        :param target_net: target network.
        """
        if len(self.experience['s']) < self.min_experiences:
            return 0

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
        loss = self.gradient_update(target_net, states, actions, rewards, states_next, ends)

        return loss

    @tf.function
    def gradient_update(self, target_net, states, actions, rewards, states_next, ends):
        """
        Gradient update with @tf.function decorator for faster performance.
        """
        # make predictions with target network and get sample q for Q-function update, sample is different if epoch ends
        target_network_enabled = True  # set False to disable target network
        double_dqn = True
        if target_network_enabled:
            if double_dqn:
                next_action = tf.math.argmax(self.predict(states_next), axis=1)
                q_values = target_net.predict(states_next)
                q_max = tf.math.reduce_sum(q_values * tf.one_hot(next_action, self.num_actions), axis=1)
            else:
                q_max = tf.math.reduce_max(target_net.predict(states_next), axis=1)
        else:
            q_max = tf.math.reduce_max(self.predict(states_next), axis=1)
        y = tf.where(ends, rewards, rewards + self.gamma * q_max)

        # perform gradient descent
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)

            # Q-values from training network for selected actions
            q_values = self.predict(states)
            selected_q_values = tf.math.reduce_sum(q_values * tf.one_hot(tf.cast(actions, tf.int32), self.num_actions), axis=1)

            loss = tf.math.reduce_sum(tf.square(y - selected_q_values))  # compute loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def get_action(self, states, epsilon=0):
        """
        Choose random action with probability 'epsilon', otherwise choose action with greedy policy, e.g. action that
        maximizes the Q-value function.
        :param states: observed states, e.g. [x, dx, th, dth].
        :param epsilon: probability of random action.
        :return: action
        """
        # take random action with probability 'epsilon'
        if np.random.random() < epsilon:
            action = np.random.choice(self.num_actions)
            return action

        # else take action that maximizes the Q-function
        else:
            q_values = self.predict(np.atleast_2d(states))
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
