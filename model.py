import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

class Model():

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

