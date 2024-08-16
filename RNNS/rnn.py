import numpy as np
from activation import Softmax,Tanh

class RNN:
    """
        this is a simple architecture of RNN implemented using python
    """

    def __init__(self,n_inputs,embd_dim) -> None:
        #initialize the learnable parameters
        self.W=np.random.uniform(n_inputs,embd_dim)
        self.U=np.random.uniform(n_inputs,embd_dim)
        self.B=np.random.uniform(n_inputs,embd_dim)
        self.V=np.random.uniform(n_inputs,embd_dim)
        self.C=np.random.uniform(n_inputs,embd_dim)

        #initialize the activation functions
        self.tanh=Tanh()

    def forward(self,prev,X):
        self.prev_a=prev
        self.x=X

        self.st=np.matmul(prev,self.W)+np.matmul(X,self.U)+self.B
        self.at=self.tanh.forward(self.st)   #this is the output hidden layer
        self.ot=np.matmul(self.V,self.at)+self.C
        #self.pred=softmax.forward(self.ot)  #turn into probabilities
        return self.ot,self.at    #raw logits & hidden unit
    
    def backward(self,d_softmax,d_loss,d_prev):
        self.dl_actua=d_softmax
        self.dl_do=d_loss
        self.dl_dv=self.dl_do*self.at
        self.dl_dc=self.dl_do

        #the section where the current and previous layer connect
        self.dl_at=self.dl_do*self.v + d_prev*self.w
        self.dl_st=self.tanh.backward(self.st,self.dl_at)

        self.dl_dw=self.dl_st*self.prev_a
        self.dl_du=self.dl_st*self.x
        self.dl_db=self.dl_st

        return self.dl_dw,self.dl_du,self.dl_db,self.dl_dv,self.dl_dc,self.dl_at
    