import numpy as np
from loss import Loss
from fnn import RNN
from activation import Softmax

class Model:
    def __init__(self,n_inputs,embd_dim,n_layers) -> None:
        #last layer
        self.softmax=Softmax()

        #initialize the learnable params
        self.W=np.random.uniform(n_inputs,embd_dim)
        self.U=np.random.uniform(n_inputs,embd_dim)
        self.B=np.random.uniform(n_inputs,embd_dim)
        self.V=np.random.uniform(n_inputs,embd_dim)
        self.C=np.random.uniform(n_inputs,embd_dim)
        
        #lets initialize the building blocks
        self.rnn=RNN()   #ALERT must be initialized correctly
        self.loss=Loss()

        #for building the neural network
        self.rnns=[]

        #lets connect multiple layer of rnns
        for _ in range(n_layers):
            self.rnns.append(self.rnn)
        
    def forward(self,inputs):
        self.prev=np.zeros_like(self.W)  #at-1
        self.preds=[]  #for accumulation

        for i in range(len(self.rnns)):
            self.logits,self.h=self.rnn[i].forward(self.u,self.W,self.W,self.B,self,self.W,inputs)
            self.probs=self.softmax(self.logits)
            self.prev=self.h
            
            #extract the most likely model predict
            self.preds.append(np.argmax(self.probs))

        return self.preds

    def backward(self):