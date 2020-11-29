import matplotlib.pyplot as plt
from data.toy_regression import ToyRegressionData
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputSpec, Dense, Wrapper, Input, concatenate, Dropout
from tensorflow.keras.models import Model
import numpy as np

TOY_DATA = "toy"
IAN_DATA = "ian"
SAMPLE_DATA = "sample"
ALLOWED_DATA_CONFIGS = {TOY_DATA, IAN_DATA, SAMPLE_DATA}

MSE = "mse"
HETEROSCEDASTIC = "heteroscedastic"
ALLOWED_LOSS_TYPES = {MSE, HETEROSCEDASTIC}


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """
    def __init__(self, layer, weight_regularizer=0, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        # initialise p
        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.random_uniform_initializer(self.init_min, self.init_max),
                                       dtype=tf.dtypes.float32,
                                       trainable=True)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x, p):
        """
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """
        eps = 1e-07
        temp = 0.1

        unif_noise = tf.random.uniform(shape=tf.shape(x))
        drop_prob = (
            tf.math.log(p + eps)
            - tf.math.log(1. - p + eps)
            + tf.math.log(unif_noise + eps)
            - tf.math.log(1. - unif_noise + eps)
        )
        drop_prob = tf.math.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=True):
        p = tf.math.sigmoid(self.p_logit)

        # initialise regulariser / prior KL term
        input_dim = inputs.shape[-1]  # last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - p)
        dropout_regularizer = p * tf.math.log(p) + (1. - p) * tf.math.log(1. - p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs, p)), regularizer
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs, p)), regularizer

            return tf.keras.backend.in_train_phase(relaxed_dropped_inputs,
                                                   self.layer.call(inputs),
                                                   training=training), regularizer


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
