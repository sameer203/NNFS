import numpy as np

class Activation:

    def forward(self, inputs):
        # Calculate Output values from inputs
        self.output = np.maximum(0, inputs)