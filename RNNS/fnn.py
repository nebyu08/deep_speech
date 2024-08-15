import numpy as np
from dataclasses import dataclass
from hidden import Layers

class RNN():
    def __init__(self
                 ,n_layers
                 ,layers:Layers
                 ) -> None:
        
        self.n_layers=n_layers
        self.layers=layers

    
    def forward(self,inputs):

        self.outputs=[]
        for i in range(self.n_layers):
