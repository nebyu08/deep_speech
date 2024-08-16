import numpy as np
import string

with open('./dataset/input.txt') as f:
    data=f.read()

char_index={char:index for index,char in enumerate(string.ascii_lowercase)}

def data_model(input_string):
    """turns string into one hot encode

    Args:
        input_string (paragraph): the input text

    Returns:
        _type_: string
    """
    paragraph=input_string.split('/n')

    index_embed=[]
    for row in range(paragraph):
        for i in row.lower():
            char_encode=[]
            char_encode=np.zeros((len(string.ascii_lowercase),1))
            char_encode[char_index[i]]=1
            index_embed.append(char_encode)
    return index_embed  

print(data_model())