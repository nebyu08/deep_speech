import numpy as np

class Softmax:
    def forward(inputs):
        """ more stabilized implementation of softmax."""
        exp_inputs=np.exp(inputs-np.max(inputs))
        return exp_inputs/np.sum(exp_inputs)
    
    def backward(actual,pred):  
        return pred-actual
