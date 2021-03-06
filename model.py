import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

class Model():

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy() 
        
        #self.loss.remember_trainable_layers(self.trainable_layers)
    
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        self.accuracy.init(y)

        train_steps=1

        if validation_data is not None :
            validation_steps = 1
            # For better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None :
            train_steps = len (X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len (X):
                train_steps += 1
            if validation_data is not None :
                validation_steps = len (X_val) // batch_size
                # Dividing rounds down. If there are some remaining
                # data, but nor full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < len (X_val):
                    validation_steps += 1

        for epoch in range(1, epochs+1):

            print ( f 'epoch: {epoch} ' )
            
            output = self.forward(X, training=True)
            data_loss, regularization_loss = self.loss.calculate(output, y, 
                                                                include_regularization=True)

            loss = data_loss + regularization_loss
            
            predictions = self.output_layer_activation.predictions(output)

            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print ( f 'epoch: {epoch} , ' +
                        f 'acc: {accuracy :.3f } , ' +
                        f 'loss: {loss :.3f } (' +
                        f 'data_loss: {data_loss :.3f } , ' +
                        f 'reg_loss: {regularization_loss :.3f } ), ' +
                        f 'lr: {self.optimizer.current_learning_rate} ' )
        
        if validation_data is not None:
            X_val, y_val = validation_data
            output = self.forward(X_val)
            loss = self.loss.calculate(output, y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print ( f 'validation, ' +
                    f 'acc: {accuracy :.3f } , ' +
                    f 'loss: {loss :.3f } ' )

    
    def forward(self, X):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)


