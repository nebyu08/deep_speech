import numpy as np

class Softmax:
    def forward(inputs):
        """ more stabilized implementation of softmax."""
        exp_inputs=np.exp(inputs-np.max(inputs))
        return exp_inputs/np.sum(exp_inputs)
    
    def backward(actual,pred):  
        return pred-actual

class Tanh:
    def forward(self,x):
        return np.tanh(x)
    
    def backward(self,x,dif_st):
        return (1-np.square(np.tanh(x)))*dif_st
    