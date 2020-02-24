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
            '10SIMELE00INACXD',
            '10SIMERI00INACXD',
            '11BRIC0000THAV0D',
            '11HEAD003STHACRA',
            '11HICR0015THACRA',
            '11HICR0015TH00RA',
            '11CHST003STHACRC',
            '11HEAD0000THACXA',
            '11HEAD0000THACRA',
            '11CHST0000THACXC',
            '11CHST0000THACRC',
            '11PELV0000THACXA',
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
            '11CLAVLEOUTHFOXA',
            '11CLAVLEINTHFOXA',
            '11ILACLE00THFOXA',
            '11ILACRI00THFOXA',
            '11NECKUP00THFOXA',
            '11NECKUP00THFOZA',
            '11NECKLO00THFOXA',
            '11NECKLO00THFOZA',
            '11HEAD003SH3ACRA',
            '11CHST003SH3ACRC',
            '11HEAD0000H3ACXA',
            '11HEAD0000H3ACRA',
            '11CHST0000H3ACXC',
            '11CHST0000H3ACRC',
            '11PELV0000H3ACXA',
            '11HICR0015H3ACRA',
            '11CHST0000H3DSXB',
            '11FEMRLE00H3FOZB',
            '11FEMRRI00H3FOZB',
            '11SEBE0000B3FO0D',
            '11SEBE0000B6FO0D',
            '11NECKUP00H3FOZA',
            '13HEAD0000HFACXA',
            '13HEAD0000HFACRA',
            '13NECKUP00HFFOXA',
            '13NECKUP00HFFOYA',
            '13NECKUP00HFFOZA',
            '13NECKUP00HFFORA',
            '13CHST0000HFACXC',
            '13CHST0000HFACRC',
            '13CHST0000HFDSXB',
            '13LUSP0000HFFOXA',
            '13ILACLE00HFFOXA',
            '13ILACRI00HFFOXA',
            '13PELV0000HFACXA',
            '13FEMRLE00HFFOZB',
            '13FEMRRI00HFFOZB',
            '13SEBE0000B3FO0D',
            '13SEBE0000B6FO0D',
            '13HEAD003SHFACRA',
            '13CHST003SHFACRC',
            '13BRIC0000HFAV0D']

table, t, chdata = initialize(directory,channels,cutoff,verbose=False)

#%% preprocessing
preprocess_channels = ['11CHSTLEUPTHDSXB','11CHSTRIUPTHDSXB','11CHSTLELOTHDSXB','11CHSTRILOTHDSXB']
chdata[preprocess_channels] = chdata[preprocess_channels].applymap(lambda x: x-x[0])
if 'TC13-006' in chdata.index:
    chdata.at['TC13-006','11SEBE0000B6FO0D'] = -chdata.at['TC13-006','11SEBE0000B6FO0D']
chdata.at['TC18-030','10SIMELE00INACXD'] = np.tile(np.nan, len(cutoff))

#%% feature extraction
def get_all_features(write_csv=False):
    i_to_t = get_i_to_t(t)
    feature_funs = {'Min_': [get_min],
                    'Max_': [get_max],
                    'Auc_': [get_auc],
                    'Tmin_': [get_argmin,i_to_t],
                    'Tmax_': [get_argmax,i_to_t]} 
    features = pd.concat(chdata.chdata.get_features(feature_funs).values(),axis=1,sort=True)
    append = [features]
    
    # get normalized values
    self_weight = table[('10','Weight')]
    partner_weight = table.loc[table[('10','Pair')],('10','Weight')]
    partner_weight.index = self_weight.index
    ratio_weight = partner_weight/self_weight
    olc_left = chdata['10SIMELE00INACXD'].apply(get_olc)
    olc_right = chdata['10SIMERI00INACXD'].apply(get_olc)
    
    olc = chdata[['10SIMELE00INACXD','10SIMERI00INACXD']].applymap(get_olc).rename(lambda x: 'OLC_' + x, axis=1)
    
    append.append(partner_weight.rename('Partner_weight'))
    append.append(ratio_weight.rename('Ratio_weight'))
    append.append(self_weight.rename('Weight'))
    append.append((self_weight*partner_weight).rename('Product_weight'))
    append.append(table[('10','Speed2')].rename('Speed2'))
    append.append(olc)
#    append.append((features.filter(regex='M[ai][xn]_')
#                           .divide(ratio_weight.loc[features.index], axis=0)
#                           .rename(lambda x: x + '/ratio_weight', axis=1)))
#    
#    append.append((features.filter(regex='M[ai][xn]_')
#                           .divide(self_weight.loc[features.index], axis=0)
#                           .rename(lambda x: x + '/weight', axis=1)))
#    
#    append.append((features.filter(regex='M[ai][xn]_')
#                           .divide(partner_weight.loc[features.index], axis=0)
#                           .rename(lambda x: x + '/partner_weight', axis=1)))
#    
#    append.append((features.filter(regex='M[ai][xn]_')
#                           .multiply(self_weight.loc[features.index], axis=0)
#                           .multiply(partner_weight.loc[features.index], axis=0)
#                           .rename(lambda x: x + '*Product_weight', axis=1)))
    
    features = pd.concat(append, axis=1)
    if write_csv:
        features.to_csv(directory + 'features.csv')
    return features

features = get_all_features(write_csv=True)

#%% get differences in features
diff_features = {}
subset = table['10'].query('Series_1==1 and Type==\'ICE\'')
diff_features['S1'] = features.loc[subset.index] - features.loc[subset['Pair']].values

subset = table['10'].query('Series_2==1').groupby('Pair_name')
diff = []
for grp in subset:
    ice_features = features.loc[grp[1].query('Type==\'ICE\'').index]
    for i in ice_features.index:
        diff.append(features.loc[[i]]-features.loc[grp[1].query('Type==\'EV\'').index].values)
diff = pd.concat(diff)
diff_features['S2'] = diff
#%%
def write_json():
    s2_ice = table.query('Series_2==1 and Type==\'ICE\' and Speed==48 and ID11==\'TH\'')
    s2_ev = s2_ice.apply(lambda x: table.query('Series_2==1 and Model==\'{}\''.format(x['Counterpart'])).index.values[0], axis=1)

    to_JSON = {'project_name': 'HEV_vs_ICE_Driver',
               'directory'   : directory,
               'cat'     : {'Series_1_ICE_48_THOR' : list(table.query('Series_1==1 and Type==\'ICE\' and Speed==48 and ID11==\'TH\'').index),
                            'Series_1_EV_48_THOR'  : list(table.query('Series_1==1 and Type==\'ICE\' and Speed==48 and ID11==\'TH\'')['Pair']),
                            'Series_1_ICE_48_H3'   : list(table.query('Series_1==1 and Type==\'ICE\' and Speed==48 and ID11==\'H3\'').index),
                            'Series_1_EV_48_H3'    : list(table.query('Series_1==1 and Type==\'ICE\' and Speed==48 and ID11==\'H3\'')['Pair']),
                            'Series_1_ICE_48_HF'   : list(table.query('Series_1==1 and Type==\'ICE\' and Speed==48 and ID13==\'HF\'').index),
                            'Series_1_EV_48_HF'    : list(table.query('Series_1==1 and Type==\'ICE\' and Speed==48 and ID13==\'HF\'')['Pair']),
                            'Series_2_ICE'         : list(s2_ev.index),
                            'Series_2_EV'          : list(s2_ev.values)},
               'data'    : [{'features'      : 'features.csv'}],
               'test'    : [{'name'       : 'Series_1_48_THOR_wilcox',
                             'test1'      : 'Series_1_ICE_48_THOR',
                             'test2'      : 'Series_1_EV_48_THOR',
                             'paired'     : True,
                             'testname'   : 'wilcoxon',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'       : 'Series_1_48_THOR_t',
                             'test1'      : 'Series_1_ICE_48_THOR',
                             'test2'      : 'Series_1_EV_48_THOR',
                             'paired'     : True,
                             'testname'   : 'ttest',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                             {'name'       : 'Series_1_48_H3_wilcox',
                             'test1'      : 'Series_1_ICE_48_H3',
                             'test2'      : 'Series_1_EV_48_H3',
                             'paired'     : True,
                             'testname'   : 'wilcoxon',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'       : 'Series_1_48_H3_t',
                             'test1'      : 'Series_1_ICE_48_H3',
                             'test2'      : 'Series_1_EV_48_H3',
                             'paired'     : True,
                             'testname'   : 'ttest',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'       : 'Series_1_48_HF_wilcox',
                             'test1'      : 'Series_1_ICE_48_HF',
                             'test2'      : 'Series_1_EV_48_HF',
                             'paired'     : True,
                             'testname'   : 'wilcoxon',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'       : 'Series_1_48_HF_t',
                             'test1'      : 'Series_1_ICE_48_HF',
                             'test2'      : 'Series_1_EV_48_HF',
                             'paired'     : True,
                             'testname'   : 'ttest',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'       : 'Series_2_48_wilcox',
                             'test1'      : 'Series_2_ICE',
                             'test2'      : 'Series_2_EV',
                             'paired'     : True,
                             'testname'   : 'wilcoxon',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'       : 'Series_2_48_t',
                             'test1'      : 'Series_2_ICE',
                             'test2'      : 'Series_2_EV',
                             'paired'     : True,
                             'testname'   : 'ttest',
                             'data'       : 'features',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'}], 
                             'test2'  : None}    
               
    with open(directory+'params.json','w') as json_file:
        json.dump(to_JSON,json_file)

if __name__=='__main__':
    write_json()