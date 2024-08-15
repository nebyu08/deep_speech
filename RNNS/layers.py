import numpy as np

class InputLayer:
    def __init__(self,n_inputs,embd_dim) -> None:
        self.n_inputs=n_inputs  #this is the number of inputs
        self.embd_dim=embd_dim  #this is the embedding dim   

        self.delta=np.random.uniform(
            low=0,
            high=1,
            size=(self.n_inputs,self.embd_dim)
        )  #shape  (30X26)

        self.bias=np.random.uniform(
            low=0,
            high=1,
            size=(self.embd_dim,
                  1)
        )  #shape (30X1)

    def _forward_input(self,inputs):
       return np.matmul(self.delta,inputs[0].T)+self.bias  #shape 30X26


class HiddenLayer:
    def __init__(self,
                 n_inputs,
                 embd_dim) -> None:
        
        self.weights=np.random.uniform(
            low=0,
            high=1,
            size=(n_inputs,embd_dim)
        )

    def _forward_input(self,prev_hidden,input_layer):
        return np.tanh(np.matmul(prev_hidden,self.weights)+input_layer)
    

class OutputLayer:
    def __init__(self,) -> None:
        self.v_weights=np.random.uniform(
         min=0,
         max=1,
         size=(1,30)   
        )

        self.c_weight=np.random.uniform(
            min=0,
            max=1,
            size=(1,)
        )

    def _forward_input(self,hidden_value_result):
        return np.matmul()