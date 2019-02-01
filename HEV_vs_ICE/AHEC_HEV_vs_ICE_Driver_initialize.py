# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:35:25 2018
a
@author: tangk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMG.read_data import initialize
from PMG.COM.get_props import *
from PMG.COM.easyname import *
import json


directory = 'P:\\Data Analysis\\Projects\\AHEC EV\\'
cutoff = range(100,1600)

channels = ['10CVEHCG0000ACXD',
            '11HEAD003STHACRA',
            '11CHST003STHACRC',
            '11HEAD0000THACXA',
            '11CHST0000THACXC',
            '11PELV0000THACXA',
            '11HICR0015THACRA',
            '11FEMRLE00THFOZB',
            '11FEMRRI00THFOZB',
            '11ACTBLE00THFORB',
            '11ACTBRI00THFORB',
            '11ACTBLE00THFOXB',
            '11ACTBRI00THFOXB',
            '11CHSTRIUPTHDSXB',
            '11CHSTRILOTHDSXB',
            '11CHSTLEUPTHDSXB',
            '11CHSTLELOTHDSXB',
            '11SPIN0100THACXC',
            '11THSP0100THAVYA',
            '11THSP0100THAVZA',
            '11SPIN1200THACXC',
            '11LUSP0000THFOXA',
            '11LUSP0000THFOYA',
            '11HEAD003SH3ACRA',
            '11CHST003SH3ACRC',
            '11HEAD0000H3ACXA',
            '11CHST0000H3ACXC',
            '11PELV0000H3ACXA',
            '11HICR0015H3ACRA',
            '11CHST0000H3DSXB',
            '11FEMRLE00H3FOZB',
            '11FEMRRI00H3FOZB',
            '11SEBE0000B3FO0D',
            '11SEBE0000B6FO0D',
            '10SIMELE00INACXD',
            '10SIMERI00INACXD',
            '11CLAVLEOUTHFOXA',
            '11CLAVLEINTHFOXA',
            '11ILACLE00THFOXA',
            '11ILACRI00THFOXA']

table, t, chdata = initialize(directory,channels,cutoff,verbose=False)

#%% preprocessing
preprocess_channels = ['11CHSTLEUPTHDSXB','11CHSTRIUPTHDSXB','11CHSTLELOTHDSXB','11CHSTRILOTHDSXB']
chdata[preprocess_channels] = chdata[preprocess_channels].applymap(lambda x: x-x[0])
if 'TC13-006' in chdata.index:
    chdata.at['TC13-006','11SEBE0000B6FO0D'] = -chdata.at['TC13-006','11SEBE0000B6FO0D']

#%% feature extraction
def get_all_features(write_csv=False):
    i_to_t = get_i_to_t(t)
    feature_funs = {'Min_': [get_min],
                    'Max_': [get_max],
                    'Tmin_': [get_argmin,i_to_t],
                    'Tmax_': [get_argmax,i_to_t]} 
    features = pd.concat(chdata.chdata.get_features(feature_funs).values(),axis=1,sort=True)
    if write_csv:
        features.to_csv(directory + 'features.csv')
    return features

features = get_all_features(write_csv=True)
#%%
to_JSON = {'project_name': 'HEV_vs_ICE_Driver',
           'directory'   : directory,
           'cat'     : {'Series_1_ICE'   : list(table.query('Series_1==1 and Type==\'ICE\'').index),
                        'Series_1_EV'    : list(table.query('Series_1==1 and Type==\'ICE\'')['Pair']),
                        'Series_2_ICE_v1': list(table.drop('TC17-025').query('Series_2==1 and Type==\'ICE\'').sort_values('Model').index),
                        'Series_2_EV_v1' : list(table.drop('TC17-025').query('Series_2==1 and Type==\'EV\'').sort_values('Counterpart').index),
                        'Series_2_ICE_v2': list(table.drop('TC15-035').query('Series_2==1 and Type==\'ICE\'').sort_values('Model').index),
                        'Series_2_EV_v2' : list(table.drop('TC15-035').query('Series_2==1 and Type==\'EV\'').sort_values('Counterpart').index)},
           'data'    : [{'control_ts'    : [directory + 'data\\', 'orig_ts']},
                        {'features'      : 'features.csv'}],
           'channels': channels,
           'cutoff'  : list(cutoff),
           'test'    : [{'name'       : 'Series_1_wilcox',
                         'test1'      : 'Series_1_ICE',
                         'test2'      : 'Series_1_EV',
                         'paired'     : True,
                         'testname'   : 'wilcoxon',
                         'data'       : 'features',
                         'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                        {'name'       : 'Series_1_t',
                         'test1'      : 'Series_1_ICE',
                         'test2'      : 'Series_1_EV',
                         'paired'     : True,
                         'testname'   : 't_test',
                         'data'       : 'features',
                         'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                        {'name'       : 'Series_2_v1_wilcox',
                         'test1'      : 'Series_2_v1_ICE',
                         'test2'      : 'Series_2_v1_EV',
                         'paired'     : True,
                         'testname'   : 'wilcoxon'},
                        {'name'       : 'Series_2_v2_wilcox',
                         'test1'      : 'Series_2_v2_ICE',
                         'test2'      : 'Series_2_v2_EV',
                         'paired'     : True,
                         'testname'   : 'wilcoxon'},
                        {'name'       : 'Series_2_v1_t',
                         'test1'      : 'Series_2_v1_ICE',
                         'test2'      : 'Series_2_v1_EV',
                         'paired'     : True,
                         'testname'   : 't_test'},
                        {'name'       : 'Series_2_v2_t',
                         'test1'      : 'Series_2_v2_ICE',
                         'test2'      : 'Series_2_v2_EV',
                         'paired'     : True,
                         'testname'   : 't_test'}], 
           'test2'  : None}

with open(directory+'params.json','w') as json_file:
    json.dump(to_JSON,json_file)
