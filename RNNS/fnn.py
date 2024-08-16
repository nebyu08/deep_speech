import numpy as np
from activation import Softmax

softmax=Softmax()

class RNN:
    def forward(self,U,W,V,B,C,prev,X):
        self.w=W
        self.v=V
        self.prev_a=prev
        self.x=X

        self.st=np.matmul(prev,W)+np.matmul(X,U)+B
        self.at=np.tanh(self.st)   #this is the output hidden layer
        self.ot=np.matmul(V,self.at)+C
        self.pred=softmax.forward(self.ot)
        return self.pred
    
    def backward(self,actual,d_prev):
        self.dl_actua=-actual/self.pred
        self.dl_do=self.pred-actual
        self.dl_dv=self.dl_do*self.at
        self.dl_dc=self.dl_do

        #the section where the current and previous layer connect
        self.dl_at=self.dl_do*self.v + d_prev*self.w
        self.dl_st=self.dl_at*(1-np.square(np.tanh(self.st),2))

        self.dl_dw=self.dl_st*self.prev_a
        self.dl_du=self.dl_st*self.x
        self.dl_db=self.dl_st

    