import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data.toy_regression import ToyRegressionData

tfkl = tf.keras.layers

TOY_DATA = "toy"
IAN_DATA = "ian"
SAMPLE_DATA = "sample"
ALLOWED_DATA_CONFIGS = {TOY_DATA, IAN_DATA, SAMPLE_DATA}

MC_DROPOUT = "mc_dropout"
CONCRETE_DROPOUT = "concrete_dropout"
DENSE = "dense"
ALLOWED_NETWORK_CONFIGS = {MC_DROPOUT, CONCRETE_DROPOUT, DENSE}

MSE = "mse"
HETEROSCEDASTIC = "heteroscedastic"
ALLOWED_LOSS_TYPES = {MSE, HETEROSCEDASTIC}


class MC_Dropout(tf.keras.Model):
    """
    Neural network with MC dropout according to
    "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
    - Gal and Ghahramani (2015): https://arxiv.org/abs/1506.02142.

    Two different models are possible depending on the specified 'loss_type':
    - 'mse': bayesian model that only predicts the output mean
    - 'heteroscedastic': bayesian model that predicts the output mean and variance; can be used to model the
    epistemic (knowledge) and aleatoric (data) uncertainty separately
    """
    def __init__(self, input_dim=1, hidden_units=[100, 100], dropout_per_layer=[0.2, 0.2], output_dim=1,
                 loss_type="mse"):
        super(MC_Dropout, self).__init__()
        
        N = 100  # data points, constant for simplicity
        lengthscale = 1e-1
        tau = 1
        reg_no_dropout = lengthscale**2.0 / (2.0 * N * tau)

        self.loss_type = loss_type

        self.input_layer = tfkl.InputLayer(input_shape=(input_dim,))
        self.hidden_layers = []
        for n_neurons, dropout_rate in zip(hidden_units, dropout_per_layer):
            reg = ((1 - dropout_rate) * lengthscale**2.0) / (2.0 * N * tau)
            self.hidden_layers.append(tfkl.Dense(n_neurons, activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(l2=reg)))
            self.hidden_layers.append(tfkl.Dropout(dropout_rate, trainable=True))
        
        self.hidden_layer_mean = tfkl.Dense(100, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.L1L2(l2=reg_no_dropout))
        self.hidden_layer_var = tfkl.Dense(100, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l2=reg_no_dropout))

        self.output_layer_mean = tfkl.Dense(output_dim, activation='linear',
                                            kernel_regularizer=tf.keras.regularizers.L1L2(l2=reg_no_dropout))
        self.output_layer_var = tfkl.Dense(output_dim, activation='linear',
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l2=reg_no_dropout))

    @tf.function
    def call(self, inputs):
        out = self.input_layer(inputs)
        for layer in self.hidden_layers:
            out = layer(out)
    
        if self.loss_type == MSE:
            # one head for the mean
            final_mean = self.output_layer_mean(out)
            return final_mean
        
        if self.loss_type == HETEROSCEDASTIC:
            # two heads for mean and variance
            y_mean = self.hidden_layer_mean(out)
            final_mean = self.output_layer_mean(y_mean)
        
            y_var = self.hidden_layer_var(out)
            final_log_var = self.output_layer_var(y_var)
        
            return final_mean, final_log_var


def plot_heteroscedastic(model, save, x_train, y_train, x_lim, y_lim):
    n_test = 500
    x_test = np.linspace(-x_lim, x_lim, n_test).reshape(n_test, 1).astype('float32')

    preds_mean = []
    preds_var = []
    n_repeats = 20
    for _ in range(n_repeats):
        pred_mean, pred_var = model(x_test, training=True)
        preds_mean.append(pred_mean)
        preds_var.append(pred_var)
    plt.figure(figsize=(10, 4))
    preds_mean = np.array(preds_mean).reshape(20, n_test)
    preds_var = np.array(preds_var).reshape(20, n_test)
    preds_mean_mean = np.mean(preds_mean, axis=0)
    preds_mean_std = np.std(preds_mean, axis=0)
    preds_var_mean = np.mean(preds_var, axis=0)

    plt.scatter(x_train, y_train, c="orangered",label='Training data')
    color_pred = (0.0, 101.0 / 255.0, 189.0 / 255.0)
    plt.plot(x_test, preds_mean_mean, color=color_pred, label='Mean function/Epistemic uncertainty')
    plt.plot(x_test, np.sqrt(np.exp(preds_var_mean)), color="green", label="Aleatoric uncertainty")
    plt.fill_between(x_test.reshape(n_test,), preds_mean_mean - preds_mean_std, preds_mean_mean + preds_mean_std,
                     alpha=0.25, color=color_pred)
    plt.fill_between(x_test.reshape(n_test,), preds_mean_mean - 2.0 * preds_mean_std, preds_mean_mean + 2.0 * preds_mean_std,
                     alpha=0.35, color=color_pred)

    plt.xlim(-x_lim, x_lim)
    plt.ylim(-y_lim, y_lim)
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig("plots/MC_Dropout_heteroscedastic.png")
    else:
        plt.show()


def plot_mse(model, save, x_train, y_train, x_lim, y_lim):
    n_test = 500
    x_test = np.linspace(-x_lim, x_lim, n_test).reshape(n_test, 1).astype('float32')

    preds = []
    n_repeats = 20
    for _ in range(n_repeats):
        pred = model(x_test, training=True)
        preds.append(pred)
    plt.figure(figsize=(10, 4))
    preds = np.array(preds).reshape(n_repeats, n_test)
    preds_mean = np.mean(preds, axis=0)
    preds_std = np.std(preds, axis=0)

    plt.scatter(x_train, y_train, c="orangered", label='Training data')
    color_pred = (0.0, 101.0 / 255.0, 189.0 / 255.0)
    plt.plot(x_test, preds_mean, color=color_pred, label='Mean function/Epistemic uncertainty')
    plt.fill_between(x_test.reshape(n_test,), preds_mean - preds_std, preds_mean + preds_std,
                     alpha=0.25, color=color_pred)
    plt.fill_between(x_test.reshape(n_test,), preds_mean - 2.0 * preds_std, preds_mean + 2.0 * preds_std,
                     alpha=0.35, color=color_pred)

    plt.xlim(-x_lim, x_lim)
    plt.ylim(-y_lim, y_lim)
    plt.legend()

    plt.tight_layout()
    if save:
        plt.savefig("plots/MC_Dropout_mse.pdf")
    else:
        plt.show()


@tf.function
def mse_loss(y_train, x_train, model):
    mse = tf.reduce_mean(tf.losses.mse(y_train, model(x_train)))
    reg = tf.reduce_sum(model.losses)  # regularization loss
    return mse + reg, reg


@tf.function
def heteroscedastic_loss(y_train, x_train, model):
    mean, log_var = model(x_train)
    mse = tf.reduce_sum(0.5 * tf.exp(-1.0 * log_var) * tf.square(y_train - mean) + 0.5 * log_var)
    reg = tf.reduce_sum(model.losses)  # regularization loss
    return mse + reg, reg


def fit_regression(loss_type="heteroscedastic", data="ian", additional_data=False, save=False):

    # load data
    if data not in ALLOWED_DATA_CONFIGS:
        raise AssertionError(f"'data' has to be in {ALLOWED_DATA_CONFIGS} but was set to {data}.")
    elif data == TOY_DATA:
        data = np.load("data/train_data_regression.npz")
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_lim, y_lim = 4.5, 70.0
    elif data == IAN_DATA:
        data = np.load("data/train_data_ian_regression.npz", allow_pickle=True)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_lim, y_lim = 12.0, 8.0
    elif data == SAMPLE_DATA:
        n_samples = 20
        toy_regression = ToyRegressionData()
        x_train, y_train = toy_regression.gen_data(n_samples)
        x_lim, y_lim = 4.5, 70.0

    if loss_type not in ALLOWED_LOSS_TYPES:
        raise AssertionError(f"'loss_type' has to be in {ALLOWED_LOSS_TYPES} but was set to {loss_type}.")
    elif loss_type == HETEROSCEDASTIC:
        y_lim = 20  # adapt y limit

    hidden_units = [100, 100]
    dropout_per_layer = [0.09, 0.119]

    model = MC_Dropout(hidden_units=hidden_units, dropout_per_layer=dropout_per_layer, loss_type=loss_type)
    
    # Add special points
    if additional_data:
        x_extension = np.array([[-10.2], [-10.1]])
        y_extension = np.array([[-6.1], [-6.2]])    
        x_train = np.insert(x_train, 0, x_extension, axis=0)
        y_train = np.insert(y_train, 0, y_extension, axis=0)
    
    epochs = 500
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, epochs, 1e-5, power=0.5)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

    for i in range(epochs):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            if loss_type == MSE:
                loss, reg = mse_loss(y_train, x_train, model)
            if loss_type == HETEROSCEDASTIC:
                loss, reg = heteroscedastic_loss(y_train, x_train, model)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        if i % int(10) == 0:
            if loss_type == "mse":
                print(f"Epoch: {i}, Loss: {loss} Regularization: {reg}")
            if loss_type == "heteroscedastic":
                print(f"Epoch: {i}, Loss: {loss} Regularization: {reg}")
            
    if loss_type == MSE:
        plot_mse(model, save, x_train, y_train, x_lim, y_lim)
    if loss_type == HETEROSCEDASTIC:
        plot_heteroscedastic(model, save, x_train, y_train, x_lim, y_lim)


if __name__ == '__main__':
    # test gpu availability
    print(f"GPU available: {tf.test.is_gpu_available()}")

    # set configuration
    loss_type = HETEROSCEDASTIC
    data = IAN_DATA  # choose from ALLOWED_DATA_CONFIGS
    additional_data = False
    save = False  # save images

    fit_regression(loss_type=loss_type, data=data, additional_data=additional_data, save=save)
