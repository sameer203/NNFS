import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

class Optimizer_SGD:

    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def update_params(self, layers):
        layers.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

    

