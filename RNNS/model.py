import numpy as np
from loss import Loss
from RNNS.rnn import RNN
from activation import Softmax

class Model:
    def __init__(self,n_inputs,embd_dim,n_layers) -> None:
        #last layer
        self.softmax=Softmax()

        #lets initialize the building blocks
        self.rnn=RNN()   #ALERT must be initialized correctly
        self.loss=Loss()

        #for building the neural network
        self.rnns=[]
        
    def forward(self,inputs):
        self.prev=np.zeros_like(self.W)  #at-1
        self.preds=[]  #for accumulation

        for i in range(len(self.rnns)):
            self.logits,self.h=self.rnns[i].forward(self.prev,inputs)
            self.probs=self.softmax(self.logits)
            self.prev=self.h
            
            #extract the most likely model predict
            self.preds.append(np.argmax(self.probs))

        return self.preds
    
    def cal_loss(self,true_value):   #loss value of single preds
        return self.loss(self.preds,true_value)
    

    def backward(self,true_label):        
        #lets initialize the gradients
        self.dl_dw=[]
        self.dl_du=[]
        self.dl_db=[]
        self.dl_dv=[]
        self.dl_dc=[]
        self.dl_at=[]  #must increment inorder to use

        for i in range(reversed(self.rnns)):
            self.dl_dw[i],self.dl_du[i],self.dl_db[i],self.dl_dv[i],self.dl_dc[i],self.dl_at[i]=self.rnns[i].backward(self.softmax.backward(true_label[i],
                                                                                                                              self.preds[i]),
                                  self.loss(true_label[i],self.preds[i]))   #the last element returend is THE gradient
    
    def update_param(self,lr):
        #lets update the params
        for i in range(len(self.rnns)):
            self.rnns[i].U=self.rnns[i].U - lr*self.dl_du
            self.rnns[i].W=self.rnns[i].W - lr*self.dl_dw
            self.rnns[i].B=self.rnns[i].B - lr*self.dl_db
            self.rnns[i].C=self.rnns[i].C - lr*self.dl_dc
            self.rnns[i].V=self.rnns[i].V - lr*self.dl_dv
    