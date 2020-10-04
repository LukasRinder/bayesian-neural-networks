"""
Toy regression problem.

Based on the toy regression task introduced in:
Hernández-Lobato et al. 2015 -
Probabilistic backpropagation for scalable learning of bayesian neural networks.
"""

import numpy as np


class ToyRegressionData():
    """
    Generates toy data for a regression task.
    """
    def __init__(self):
        self.x_lim = [-4, 4]
        self.sigma = 3
        self.eps_loc = 0.0
        self.eps_scale = 1.0

    def gen_data(self, n_samples):
        x = np.random.uniform(self.x_lim[0], self.x_lim[1], size=(n_samples, 1)).astype('float32')
        epsilon = np.random.normal(self.eps_loc, self.eps_scale, size=x.shape).astype('float32')
        y = np.power(x, 3) + self.sigma * epsilon

        return x, y

    def eval_data(self, x):
        return np.power(x, 3)
