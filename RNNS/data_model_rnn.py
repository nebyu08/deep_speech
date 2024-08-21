import numpy as np
import string

char_index={char:index for index,char in enumerate(string.ascii_lowercase)}

def data_model(input_string):
    """turns string into one hot encode

    Args:
        input_string (paragraph): the input text

    Returns:
        _type_: string
    """

    assert isinstance(input_string,string)

    index_embed=[]
    for word in input_string:
        for char in word:
            char=char.lower()
            char_encode=[]
            char_encode=np.zeros((len(string.ascii_lowercase),1))
            char_encode[char_index[char]]=1
            index_embed.append(char_encode)
            
    return index_embed  
