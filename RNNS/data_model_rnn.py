import numpy as np
import re
import string


def data_model(input_string):
    """turns string into one hot encode

    Args:
        input_string (paragraph): the input text

    Returns:
        _type_: string
    """

    #creating new charcher to index
    char_index={char:index for index,char in enumerate(string.ascii_lowercase)}
    char_index['']=26

    assert isinstance(input_string,str)

    #lets remove new line charachchter

    input_string=re.sub(r'\n','',input_string)

    index_embed=[]
    for word in input_string:
        char_encode=[]
        for char in word:
            char=char.lower()
            char_vector=np.zeros((len(string.ascii_lowercase),1))
            char_vector[char_index[char]]=1
            char_encode.append(char_vector)
        index_embed.append(char_encode)

    return index_embed  
