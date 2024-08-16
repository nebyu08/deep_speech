import numpy as np
from activation import Softmax

softmax=Softmax()


class RNN:
    def forward(self,U,W,V,B,C,prev,X):
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
        
