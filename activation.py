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

# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Create activation and loss function objects
    def _init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the outputs
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # Number of sample  
        samples = len(dvalues)
        # If labels are one-hot encoded turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Calculate Gradients
        self.dinput[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs/samples 

class Activation_Sigmoid():

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


