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

#%%
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import StandardScaler
y = diff_features['Min_11HEAD0000THACXA']
x = diff_features.drop([i for i in diff_features.columns if y.name in i], axis=1)
drop = [i for i in x.columns if '11' in i or '13' in i or '/' in i or '*' in i] 

x = x.drop(drop, axis=1)
x = x.loc[~y.isna()]
x = x.dropna(axis=1)
y = y.dropna()

ss = StandardScaler()
x = pd.DataFrame(ss.fit_transform(x), index=x.index, columns=x.columns)

model = LassoLars()
model = model.fit(x, y)
coefs = pd.Series(model.coef_, index=x.columns)
coefs = coefs[coefs.abs()>0]
print(coefs)