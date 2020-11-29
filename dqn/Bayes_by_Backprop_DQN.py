import numpy as np
import tensorflow as tf

from bayes.Bayes_by_Backprop import BayesByBackprop

tfkl = tf.keras.layers


class BBB_Backbone(tf.keras.Model):
    """
    Backbone of the Deep Q-Network (DQN) with Bayes by Backprop - Blundell et al. (2015).

    Takes 'num_states' inputs and outputs one Q-value for each action.
    """
    def __init__(self, num_states, hidden_units, num_actions, max_std=1.0, log_var_mean_init=-3.0, log_var_init=1e-3):
        super(BBB_Backbone, self).__init__()
        self.input_layer = tfkl.InputLayer(input_shape=(num_states,))

        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tfkl.Dense(i, activation='relu', kernel_initializer='RandomNormal'))
        self.dense_bbb_out = BayesByBackprop(n_out=num_actions, max_std=max_std, log_var_mean_init=log_var_mean_init,
                                             log_var_init=log_var_init)

    @tf.function
    def call(self, inputs, same_noise=False, training=True):
        out = self.input_layer(inputs)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.dense_bbb_out(out, same_noise=same_noise, training=training)
        return out

    def kl_div(self, same_noise=True):
        """
        Compute current KL divergence of the Bayes by Backprop layers.
        Used as a regularization term during training.
        """
        kldiv = self.dense_bbb_out.kl_div(same_noise)
        return kldiv

    def reset_noise(self):
        """
        Re-sample noise/epsilon parameters of the Bayes by Backprop layers. Required for the case of having the same
        epsilon parameters across one batch.
        """
        self.dense_bbb_out.reset_noise()

    def print_variance(self):
        print(f"Variance layer 1: {self.hidden_layers[0].log_var_W}")


class BBBDQN(tf.Module):
    """
    Deep Q-Network utilizing Bayes by Backprop for efficient sampling.
    """
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr,
                 alpha):
        super(BBBDQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.kl_coeff = alpha*batch_size / max_experiences
        self.model = BBB_Backbone(num_states, hidden_units, num_actions, max_std=0.5, log_var_mean_init=-3.0,
                                  log_var_init=1e-3)
        self.experience = {'s': [], 'a': [], 'r': [], 's_next': [], 'end': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs, same_noise=False, training=True):
        """
        Get Q-values from backbone network.
        :param inputs: inputs for the backbone network, e.g. states.
        :param same_noise: uses the same epsilon parameter for one mini-batch, if set to `True`.
        :param training: forward pass without stochasticity, if set to `False`.
        :return: outputs of the backbone network, e.g. num_action Q-values.
        """
        return self.model(tf.convert_to_tensor(inputs, tf.float32), same_noise=same_noise, training=training)

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
        loss, kl_loss = self.gradient_update(target_net, states, actions, rewards, states_next, ends)

        return loss, kl_loss

    @tf.function
    def gradient_update(self, target_net, states, actions, rewards, states_next, ends):
        """
        Gradient update with @tf.function decorator for faster performance.
        """
        # make predictions with target network without stochasticity and get sample q for Q-function update
        # sample is different if epoch ends
        double_dqn = True
        if double_dqn:
            next_action = tf.math.argmax(self.predict(states_next, training=False), axis=1)
            q_values = target_net.predict(states_next, training=False)
            q_max = tf.math.reduce_sum(q_values * tf.one_hot(next_action, self.num_actions), axis=1)
        else:
            q_max = tf.math.reduce_max(target_net.predict(states_next, training=False), axis=1)

        y = tf.where(ends, rewards, rewards + self.gamma * q_max)

        self.model.reset_noise()  # sample new epsilon_w and epsilon_z

        # perform gradient descent
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)

            kl_loss = self.kl_coeff * self.model.kl_div(same_noise=True)
            # Q-values from training network for selected actions
            q_values = self.predict(states, same_noise=True)
            selected_q_values = tf.math.reduce_sum(q_values * tf.one_hot(tf.cast(actions, tf.int32), self.num_actions),
                                                   axis=1)

            td_error = tf.math.reduce_sum(tf.square(y - selected_q_values))
            loss = td_error + kl_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.model.reset_noise()  # sample new epsilon_w and epsilon_z

        return loss, kl_loss

    def get_action(self, states, same_noise=False, training=True):
        """
        Predict action with the Bayes By Backprop network. In each forward pass the weights are sampled from the weight
        posterior distribution. Hence, approximated Thompson sampling is performed. For uncertain weight posterior
        distributions the variance in the sampled values will be higher, leading inherently to more exploration.

        :param states: observed states, e.g. [x, dx, th, dth].
        :return: action
        """
        q_values = self.predict(np.atleast_2d(states), same_noise=same_noise, training=training)
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
