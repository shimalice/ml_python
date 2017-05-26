# coding:utf-8
import numpy as np

class Perceptron(object):
    """docstring for Perceptron.

    パラメータ
    eta : float, learning rate 0.0-1.0
    n_iter : int, the number of Training

    attribute
    w_ : Array 1dimention, weight
    errors_ : list, each epoc errors"""
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Training data

        parameter
        X : shape = [n_samples, n_features], Training data
        y : shape = [n_samples] Cost function

        return
        self : object"""
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
