import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

class Dropout():

    def __init__(self, rate):
        self.rate = 1 - rate
    
    def forward(self,inputs):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=input.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
        