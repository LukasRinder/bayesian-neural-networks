import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.toy_regression import ToyRegressionData
from bayes.ConcreteDropout import ConcreteDropout
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputSpec, Dense, Wrapper, Input, concatenate
from tensorflow.keras.models import Model


TOY_DATA = "toy"
IAN_DATA = "ian"
SAMPLE_DATA = "sample"
ALLOWED_DATA_CONFIGS = {TOY_DATA, IAN_DATA, SAMPLE_DATA}

MSE = "mse"
HETEROSCEDASTIC = "heteroscedastic"
ALLOWED_LOSS_TYPES = {MSE, HETEROSCEDASTIC}


def mse_loss(true, pred):
    return tf.reduce_mean((true - pred) ** 2, -1)


def heteroscedastic_loss(y_train, pred):
    n_outputs = pred.shape[1] // 2
    mean = pred[:, :n_outputs]
    log_var = pred[:, n_outputs:]
    return tf.reduce_sum(0.5 * tf.exp(-1 * log_var) * tf.square(y_train - mean) + 0.5 * log_var)


def make_model(loss_type, n_features, n_outputs, n_nodes=400, dropout_reg=1e-5, wd=1e-3):
    losses = []
    inp = Input(shape=(n_features,))
    x = inp
        
    x, loss = ConcreteDropout(Dense(n_nodes, activation='relu'),
                              weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    x, loss = ConcreteDropout(Dense(n_nodes, activation='relu'),
                              weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    x, loss = ConcreteDropout(Dense(n_nodes, activation='relu'),
                              weight_regularizer=wd, dropout_regularizer=dropout_reg)(x)
    losses.append(loss)
    
    if loss_type == MSE:
        mean = Dense(100, activation='relu')(x)
        final_mean = Dense(n_outputs, activation='linear')(mean)
        model = Model(inp, final_mean)
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 500, 1e-5, power=0.5)
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate_fn), loss=mse_loss)
        
    if loss_type == HETEROSCEDASTIC:
        mean = Dense(100, activation='relu')(x)
        final_mean = Dense(n_outputs, activation='linear')(mean)
    
        log_var = Dense(100, activation='relu')(x)
        final_log_var = Dense(n_outputs, activation='linear')(log_var)
    
        out = concatenate([final_mean, final_log_var])
        model = Model(inp, out)
        for loss in losses:
            model.add_loss(loss)
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-3, 500, 1e-5, power=0.5)
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate_fn), loss=heteroscedastic_loss,
                      metrics=[mse_loss])

    return model


def plot_heteroscedastic(model, save, x_train, y_train, x_lim, y_lim):
    n_test = 500
    x_test = np.linspace(-x_lim, x_lim, n_test).reshape(n_test, 1).astype('float32')

    preds_mean = []
    preds_var = []
    n_repeats = 20
    for _ in range(n_repeats):
        pred = model(x_test, training=True)
        n_outputs = pred.shape[1] // 2
        pred_mean = pred[:, :n_outputs]
        pred_var = pred[:, n_outputs:]
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
        plt.savefig("plots/Concrete_Dropout_heteroscedastic.png")
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
        plt.savefig("plots/Concrete_Dropout_mse.pdf")
    else:
        plt.show()


def fit_regression(loss_type="heteroscedastic", data="ian", save=False):
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

    n_epochs = 500
    l = 1e-3  # length-scale
    weight_reg = l**2.0 / len(x_train)
    dropout_reg = 2.0 / len(x_train)

    model = make_model(loss_type, 1, 1, n_nodes=200, dropout_reg=dropout_reg, wd=weight_reg)

    print("Starting training...")
    model.fit(x_train, y_train, epochs=n_epochs)

    print("Starting plotting...")
    if loss_type == "mse":
        plot_mse(model, save, x_train, y_train, x_lim, y_lim)
    if loss_type == "heteroscedastic":
        plot_heteroscedastic(model, save, x_train, y_train, x_lim, y_lim)

    print("Dropout rates:")
    for i in model.layers:
        if isinstance(i, ConcreteDropout):
            print(tf.math.sigmoid(i.p_logit))


if __name__ == '__main__':
    # test gpu availability
    print(f"GPU available: {tf.test.is_gpu_available()}")

    # set configuration
    loss_type = MSE
    data = IAN_DATA  # choose from ALLOWED_DATA_CONFIGS
    save = False  # save images

    fit_regression(loss_type=loss_type, data=data, save=save)
