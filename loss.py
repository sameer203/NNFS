import numpy as np
import nnfs


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
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Probabilities for target values only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]

        # Mask values only for One Hot Encoded labels
        elif len(y_true.shape) ==2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of sample  
        samples = len(dvalues)

        # No of labels in each sample
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels[y_true])

        # Calculate gradients
        self.dinputs = -y_true/dvalues

        # Normalize Gradients
        self.dinputs = self.dinputs/samples


        
