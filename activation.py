import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        # Calculate Output values from inputs
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    def forward(self,inputs):

        # Get Normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalizethem for each samples
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities