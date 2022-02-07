import numpy as np


class Accuracy():
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.prediction is None or reinit:
            self.precision = np.std(y)/250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision