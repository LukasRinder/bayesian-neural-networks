import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data.toy_regression import ToyRegressionData
from bayes.MNF import DenseMNF
from bayes.Bayes_by_Backprop import BayesByBackprop

tfkl = tf.keras.layers

TOY_DATA = "toy"
IAN_DATA = "ian"
SAMPLE_DATA = "sample"
ALLOWED_DATA_CONFIGS = {TOY_DATA, IAN_DATA, SAMPLE_DATA}

MNF = "mnf"
BAYES_BY_BACKPROP = "bayesbybackprop"
DENSE = "dense"
ALLOWED_NETWORK_CONFIGS = {MNF, BAYES_BY_BACKPROP, DENSE}


class MLP(tf.Module):
    """
    Simple Multi-layer Perceptron Model.
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.input_layer = tfkl.InputLayer(input_shape=(1,))
        self.hidden_layer_1 = tfkl.Dense(100, activation='relu')
        self.hidden_layer_2 = tfkl.Dense(100, activation='relu')
        self.output_layer = tfkl.Dense(1, activation='linear')

    @tf.function
    def __call__(self, x, *args, **kwargs):
        y = self.input_layer(x)
        y = self.hidden_layer_1(y)
        y = self.hidden_layer_2(y)
        y = self.output_layer(y)
        return y


class BNN_MNF(tf.Module):
    """
    Bayesian Neural Network with fully-connected layers utilizing Multiplicative Normalizing Flows by Christos Louizos, Max Welling
    (Jun 2017).
    """
    def __init__(self, input_dim=1, hidden_units=[100, 100], output_dim=1, hidden_bayes=False, use_z=True, max_std=1.0):
        super(BNN_MNF, self).__init__()
        self.input_layer = tfkl.InputLayer(input_shape=(input_dim,))

        self.hidden_layers = []
        self.hidden_bayes = hidden_bayes
        for i in hidden_units:
            if self.hidden_bayes:
                self.hidden_layers.append(DenseMNF(n_out=i, use_z=use_z, max_std=max_std))
            else:
                self.hidden_layers.append(tfkl.Dense(i, activation='relu', kernel_initializer='RandomNormal'))

        self.dense_mnf_out = DenseMNF(n_out=output_dim, use_z=use_z, max_std=max_std)

    @tf.function
    def __call__(self, inputs, same_noise=False, training=True, *args, **kwargs):
        out = self.input_layer(inputs)
        for layer in self.hidden_layers:
            if self.hidden_bayes:
                out = layer(out, same_noise=same_noise, training=training)
                out = tf.nn.relu(out)
            else:
                out = layer(out)  # relu already in keras layer
        out = self.dense_mnf_out(out, same_noise=same_noise, training=training)

        return out

    def kl_div(self, same_noise=True):
        """
        Compute current KL divergence of all layers.
        Can be used as a regularization term during training.
        """
        kldiv = 0
        if self.hidden_bayes:
            for dense_mnf in self.hidden_layers:
                kldiv = kldiv + dense_mnf.kl_div(same_noise)
        kldiv = kldiv + self.dense_mnf_out.kl_div(same_noise)
        return kldiv

    def reset_noise(self):
        if self.hidden_bayes:
            for dense_mnf in self.hidden_layers:
                dense_mnf.reset_noise()
        self.dense_mnf_out.reset_noise()


class BNN_BBB(tf.Module):
    """
    Bayesian Neural Network with fully-connected layers utilizing Bayes by Backprop by Blundell et al. (2015).
    """
    def __init__(self, input_dim=1, hidden_units=[100, 100], output_dim=1, hidden_bayes=False, max_std=1.0):
        super(BNN_BBB, self).__init__()
        self.input_layer = tfkl.InputLayer(input_shape=(input_dim,))

        self.hidden_layers = []
        self.hidden_bayes = hidden_bayes
        for i in hidden_units:
            if hidden_bayes:
                self.hidden_layers.append(BayesByBackprop(n_out=i, max_std=max_std))
            else:
                self.hidden_layers.append(tfkl.Dense(i, activation='relu', kernel_initializer='RandomNormal'))
        self.dense_bbb_out = BayesByBackprop(n_out=output_dim, max_std=max_std)

    @tf.function
    def __call__(self, inputs, same_noise=False, training=True, *args, **kwargs):
        out = self.input_layer(inputs)
        for layer in self.hidden_layers:
            if self.hidden_bayes:
                out = layer(out, same_noise=same_noise, training=training)
                out = tf.nn.relu(out)
            else:
                out = layer(out)  # relu already in keras layer
        out = self.dense_bbb_out(out, same_noise=same_noise, training=training)
        return out

    def kl_div(self, same_noise=True):
        """
        Compute current KL divergence of the Bayes by Backprop layers.
        Used as a regularization term during training.
        """
        kldiv = 0
        if self.hidden_bayes:
            for dense_bbb in self.hidden_layers:
                kldiv = kldiv + dense_bbb.kl_div(same_noise)
        kldiv = kldiv + self.dense_bbb_out.kl_div(same_noise)
        return kldiv

    def reset_noise(self):
        """
        Re-sample noise/epsilon parameters of the Bayes by Backprop layers. Required for the case of having the same
        epsilon parameters across one batch.
        """
        if self.hidden_bayes:
            for dense_bbb in self.hidden_layers:
                dense_bbb.reset_noise()
        self.dense_bbb_out.reset_noise()


@tf.function
def loss_fn(y_train, x_train, model, bayes, reg=1.0, same_noise=False):
    if bayes:
        # divide by divided by the total number of samples in an epoch (batch_size * steps_per_epoch)
        # here: steps_per_epoch = 1
        mse = tf.reduce_mean(tf.losses.mse(y_train, model(x_train, same_noise=same_noise)))
        kl_loss = model.kl_div() / tf.cast(x_train.shape[0]*reg, tf.float32)
    else:
        mse = tf.reduce_mean(tf.losses.mse(y_train, model(x_train)))
        kl_loss = 0

    return mse + kl_loss, kl_loss


def fit_regression(network, hidden_bayes=False, same_noise=False, max_std=0.5, data="ian", save=False):

    # load data
    if data not in ALLOWED_DATA_CONFIGS:
        raise AssertionError(f"'data' has to be in {ALLOWED_DATA_CONFIGS} but was set to {data}.")
    elif data == TOY_DATA:
        data = np.load("data/train_data_regression.npz")
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_lim, y_lim = 4.5, 70.0
        reg = 10.0  # regularization parameter lambda
    elif data == IAN_DATA:
        data = np.load("data/train_data_ian_regression.npz", allow_pickle=True)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_lim, y_lim = 12.0, 8.0
        reg = 30  # regularization parameter lambda
    elif data == SAMPLE_DATA:
        n_samples = 20
        toy_regression = ToyRegressionData()
        x_train, y_train = toy_regression.gen_data(n_samples)
        x_lim, y_lim = 4.5, 70.0
        reg = 10.0  # regularization parameter lambda

    # choose network
    if network not in ALLOWED_NETWORK_CONFIGS:
        raise AssertionError(f"'network' has to be in {ALLOWED_NETWORK_CONFIGS} but was set to {network}.")
    elif network == MNF:
        model = BNN_MNF(hidden_bayes=hidden_bayes, max_std=max_std)
        bayes = True
    elif network == BAYES_BY_BACKPROP:
        model = BNN_BBB(hidden_bayes=hidden_bayes, max_std=max_std)
        bayes = True
    elif network == DENSE:
        model = MLP()
        bayes = False

    epochs = 500
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, epochs, 1e-6, power=0.5)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    # initialize
    _, _ = loss_fn(y_train, x_train, model, bayes, reg, same_noise)

    train_losses = []
    kl_losses = []
    for i in range(epochs):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            loss, kl_loss = loss_fn(y_train, x_train, model, bayes, reg, same_noise)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        if same_noise:
            model.reset_noise()  # sample new epsilons

        train_losses.append(loss)
        kl_losses.append(kl_loss)

        if i % int(10) == 0:
            print(f"Epoch: {i}, MSE: {loss}, KL-loss: {kl_loss}")

    plt.plot(range(epochs), train_losses)
    plt.plot(range(epochs), kl_losses)
    plt.legend(["Train loss", "KL loss"])

    n_test = 500
    x_test = np.linspace(-x_lim, x_lim, n_test).reshape(n_test, 1).astype('float32')

    if bayes:
        y_preds = []
        for _ in range(20):
            y_pred = model(x_test)
            y_preds.append(y_pred)
        plt.figure(figsize=(10, 4))
        y_preds = np.array(y_preds).reshape(20, n_test)
        y_preds_mean = np.mean(y_preds, axis=0)
        y_preds_std = np.std(y_preds, axis=0)

        plt.scatter(x_train, y_train, c="orangered")
        color_pred = (0.0, 101.0 / 255.0, 189.0 / 255.0)
        plt.plot(x_test, y_preds_mean, color=color_pred)
        plt.fill_between(x_test.reshape(n_test,), y_preds_mean - y_preds_std, y_preds_mean + y_preds_std,
                         alpha=0.25, color=color_pred)
        plt.fill_between(x_test.reshape(n_test,), y_preds_mean - 2.0 * y_preds_std, y_preds_mean + 2.0 * y_preds_std,
                         alpha=0.35, color=color_pred)

        plt.xlim(-x_lim, x_lim)
        plt.ylim(-y_lim, y_lim)
        plt.legend(["Mean function", "Observations"])

    else:
        plt.figure(figsize=(10, 4))
        y_pred = model(x_test)
        plt.scatter(x_train, y_train, c="orangered")
        color_pred = (0.0, 101.0 / 255.0, 189.0 / 255.0)
        plt.plot(x_test, y_pred, color=color_pred)
        plt.xlim(-x_lim, x_lim)
        plt.ylim(-y_lim, y_lim)
        plt.legend(["Mean function", "Observations"])

    plt.tight_layout()
    if save:
        plt.savefig(f"plots/{network}.pdf")
    else:
        plt.show()


if __name__ == '__main__':
    # test gpu availability
    print(f"GPU available: {tf.test.is_gpu_available()}")

    # set configuration
    network = MNF  # choose from ALLOWED_NETWORK_CONFIGS
    hidden_bayes = False  # False: last layer bayes, True: all layers bayes
    same_noise = True  # set if same noise/epsilon should be used within a batch
    max_std = 0.5
    data = IAN_DATA  # choose from ALLOWED_DATA_CONFIGS
    save = False  # save images

    fit_regression(network=network, hidden_bayes=hidden_bayes, same_noise=same_noise, max_std=max_std, data=data,
                   save=save)
