import numpy as np

def xavier_normal_initial(input_dim,output_dim):
    stdv=np.sqrt(2/(input_dim+output_dim))
    return np.random.uniform(0,stdv,(input_dim,output_dim))