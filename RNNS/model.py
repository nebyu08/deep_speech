import numpy as np
from loss import Loss
from fnn import RNN
from activation import Softmax

class Model:
    def __init__(self,n_inputs,embd_dim) -> None:
        #initialize the learnable params
        self.W=np.random.uniform(n_inputs,embd_dim)
        self.U=np.random.uniform(n_inputs,embd_dim)
        self.B=np.random.uniform(n_inputs,embd_dim)
        self.V=np.random.uniform(n_inputs,embd_dim)
        self.C=np.random.uniform(n_inputs,embd_dim)
        
        #lets initialize the building blocks
        self.rnn=RNN()
        self.activation=Softmax()
        self.loss=Loss()
        
    def forward(self,inputs):
        