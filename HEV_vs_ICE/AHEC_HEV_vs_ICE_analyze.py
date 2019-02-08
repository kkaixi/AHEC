# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:06:50 2019

@author: tangk
"""

import json
import pandas as pd
import numpy as np

directory = 'P:\\Data Analysis\\Projects\\AHEC EV\\'
with open(directory + 'params.json','r') as json_file:
    params = json.load(json_file)

res = {}
for data in params['test']:
    label1 = data['label1']
    label2 = data['label2']
    index = pd.MultiIndex.from_arrays([label1,label2])
    res[data['name'][0]] = pd.Series(data['res'],index=index).unstack().dropna(axis=0, how='all').astype(np.float32)
    
for test, data in res.items():
    print(test)
    print(data['p'][data['p']<0.05])
