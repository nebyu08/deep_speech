import numpy as np

class Loss:
    """the loss here is Cross Entropy Loss"""
    def forward(self,pred,actual):
        return -actual*np.log(pred)
    
    def backward(self,actual,pred):
        return -actual/pred