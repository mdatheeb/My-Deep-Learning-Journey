import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))


class nueron:
    def __init__(self, weight, bias):
        self.bias = bias
        self.weight = weight

    def feedforward(self, inputs):
        total = np.dot(self.weight, inputs) + self.bias
        return sigmoid(total)


weight = np.array([0, 1])
bias = 4
n = nueron(weight, bias)


x = np.array([2, 3])
print(n.feedforward(x))
