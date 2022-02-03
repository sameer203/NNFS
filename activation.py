import numpy as np

class Activation_ReLU:

    def forward(self, inputs):

        # Rembember input values
        self.inputs = inputs
        # Calculate Output values from inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # copy of the values
        self.dinputs = dvalues.copy()

        # Zero gradients where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    def forward(self,inputs):

        # Get Normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalizethem for each samples
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        # Create Unintitalized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate output and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten Output array
            single_output = single_output.shape(-1,1)
            # Calculate the Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.inputs[index] = np.dot(jacobian_matrix, single_dvalues)