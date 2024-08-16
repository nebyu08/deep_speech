import numpy as np
import string

with open('./dataset/input.txt') as f:
    data=f.read()

char_index={char:index for index,char in enumerate(string.ascii_lowercase)}

def data_model():
    index_embed=[]
    for i in range(26):
        char_encode=[]
        char_encode=np.zeros(26)
        char_encode[i]=1
        index_embed.append(char_encode)
    return index_embed  

print(data_model())