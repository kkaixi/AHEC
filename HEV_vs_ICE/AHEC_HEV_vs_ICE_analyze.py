# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:06:50 2019

@author: tangk
"""

import json
import pandas as pd
import numpy as np
import seaborn as sns

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

#%% create a linear regression model and use it to predict responses based on weight/partner weight to 
# assess how much the two account for observed differences in the response
from sklearn.linear_model import LinearRegression
#ylist = ['Max_11HEAD003STHACRA/partner_weight',
#         'Max_11CHST003STHACRC/partner_weight',
#         'Max_11CHST003SH3ACRC/partner_weight']
#xlist = ['Weight']
ylist = ['Max_11HEAD003STHACRA',
         'Max_11CHST003STHACRC',
         'Max_11CHST003SH3ACRC']
xlist = ['Weight']
error = []
subset = table.drop('TC12-006').query('Type==\'ICE\'')
subset_test = table.drop('TC12-006')
for chx in xlist:
    for chy in ylist:
        x = features.loc[subset.index, chx]
        y = features.loc[subset.index, chy]
        i = ~(x.isna() | y.isna())
        i = i[i].index
        x, y = x[i].to_frame().values, y[i].to_frame().values
        
        x_test = features.loc[subset_test.index, chx]
        y_test = features.loc[subset_test.index, chy]
        i_test = ~(x_test.isna() | y_test.isna())
        i_test = i_test[i_test].index
        x_test, y_test = x_test[i_test].to_frame().values, y_test[i_test].to_frame().values
        y_test = np.squeeze(y_test)
        
        lr = LinearRegression()
        lr = lr.fit(x, y)
        y_pred = np.squeeze(lr.predict(x_test))
        err = y_pred-y_test
        error.append(pd.DataFrame({'Error': err, 'TC': i_test, 'Response': chy, 'Type': table.loc[i_test, 'Type']}))
error = pd.concat(error)

#ax = sns.boxplot(x='Response', y='Error', hue='Type', boxprops={'alpha': 0.5}, data=error)
ax.axhline(0,linewidth=1,color='k')
ax = sns.stripplot(x='Response', y='Error', hue='Type', data=error, ax=ax)
ax.tick_params(axis='x', rotation=90)