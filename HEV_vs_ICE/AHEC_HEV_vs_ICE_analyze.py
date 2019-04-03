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
    
#for test, data in res.items():
#    pvals = data['p'].filter(regex='M[ai][xn]_\w{16}$')
#    indices_sig = pvals[pvals<0.05]
#    print(test)
#    print(indices_sig)
    
plot_channels = ['Min_10CVEHCG0000ACXD',
                 'Min_10SIMELE00INACXD',
                 'Max_11HEAD0000THACRA',
                 'Max_11HEAD0000H3ACRA',
                 'Min_11HEAD0000THACXA',
                 'Min_11HEAD0000H3ACXA',
                 'Max_11NECKUP00THFOZA',
                 'Max_11NECKUP00H3FOZA',
                 'Max_11CHST0000THACRC',
                 'Max_11CHST0000H3ACRC',
                 'Min_11CHST0000THACXC',
                 'Min_11CHST0000H3ACXC',
                 'Min_11PELV0000THACXA',
                 'Min_11PELV0000H3ACXA',
                 'Min_11SPIN0100THACXC',
                 'Max_13HEAD0000HFACRA',
                 'Min_13HEAD0000HFACXA',
                 'Max_13NECKUP00HFFOZA',
                 'Max_13CHST0000HFACRC',
                 'Min_13CHST0000HFACXC',
                 'Min_13PELV0000HFACXA']
for k, df in res.items():
    print(k)
    print(df.loc[plot_channels, 'p'].dropna(axis=0, how='all'))

