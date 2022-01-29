import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self,inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

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

class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values

    def calculate(self, output, y):

        # Calculate sample loss
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return Loss
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        
        # Number of samples in each bactch
        samples = len(y_pred)

        # Clip the data to prevent division by 0
        # Clip both sides to not to drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]

        # Mask values only for One Hot Encoded labels
        elif len(y_true.shape) ==2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods



# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)

print('loss:', loss)
