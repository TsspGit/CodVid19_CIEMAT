__author__ = '@Tssp'

import numpy as np
import pandas as pd

def map_categorical(row, values_dict):
    '''This function transforms a categorical string 
    like "pneumonia", "CODVID-19" and "normal" to an
    integer categorical variable. Adds the new integer
    column to the dataset.
    Example: 
    values_dict = {'pneumonia': 0, 'COVID-19': 1, 'normal': 2}
    test_class['class_categorical'] = test_class['class'].apply(map_categorical, args=(values_dict,))
    '''
    return values_dict[row]