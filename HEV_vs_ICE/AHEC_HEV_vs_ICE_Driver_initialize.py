# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:35:25 2018

@author: tangk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMG.read_data import initialize
from PMG.COM.get_props import peakval, get_ipeak, get_tonset, get_i2peak, get_t2peak
from PMG.COM.easyname import renameISO, rename_list
#from PMG.COM.arrange import sep_by_peak
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
drop = ['TC13-217',
        'TC14-012',
        'TC14-214',
        'TC12-013',
        'TC11-007']

table, t, chdata = initialize(directory,channels,cutoff,drop=drop,verbose=False)

#%% preprocessing
preprocess_channels = ['11CHSTLEUPTHDSXB','11CHSTRIUPTHDSXB','11CHSTLELOTHDSXB','11CHSTRILOTHDSXB']
chdata[preprocess_channels] = chdata[preprocess_channels].applymap(lambda x: x-x[0])
if 'TC13-006' in chdata.index:
    chdata.at['TC13-006','11SEBE0000B6FO0D'] = -chdata.at['TC13-006','11SEBE0000B6FO0D']

#%% feature extraction
mins, maxs = sep_by_peak(chdata.applymap(peakval))
tmins, tmaxs = sep_by_peak(chdata.applymap(get_i2peak))
tmins = get_t2peak(t, tmins)

#%% save features
mins = mins.rename(lambda name: (name+'_Min'),axis=1).dropna(axis=1,how='all')
maxs = maxs.rename(lambda name: (name+'_Max'),axis=1).dropna(axis=1,how='all')
tmins = tmins.rename(lambda name: (name+'_Tmin'),axis=1).dropna(axis=1,how='all')

features = {}
for dummy in ['TH','H3']:
    features[dummy] = []
    for df in [mins,maxs,tmins]:
        ch = [i for i in df.columns if (i[:2]=='10') or (i[10:12]==dummy)]
        features[dummy].append(df[ch])
    features[dummy] = pd.concat(features[dummy],axis=1).rename_axis('TC')

#%% save features and json to file
if __name__=='__main__':
    for dummy in ['TH','H3']:
        features[dummy].to_csv(directory + 'features_' + dummy + '.csv')
    #%%
    to_JSON = {'project_name': 'HEV_vs_ICE_Driver',
               'directory'   : directory,
               'cat'     : {'control'     : list(table.query('ID11==\'TH\' and Subset==\'CONTROL\'').index)[0::2]+list(table.query('Subset==\'CONTROL\'').index)[1::2],
                            'control_pair': list(table.query('ID11==\'TH\' and Subset==\'CONTROL\'').index)[1::2]+list(table.query('Subset==\'CONTROL\'').index)[0::2],
                            'THOR'        : list(table.query('ID11==\'TH\' and Subset==\'HEV vs ICE\'').index[0::2]),
                            'THOR_pair'   : list(table.query('ID11==\'TH\' and Subset==\'HEV vs ICE\'').index[1::2]),
                            'H3'          : list(table.query('ID11==\'H3\' and Subset==\'HEV vs ICE\'').index[0::2]),
                            'H3_pair'     : list(table.query('ID11==\'H3\' and Subset==\'HEV vs ICE\'').index[1::2]),
                            'double_peak' :   ['TC17-505','TC17-012','TC14-035','TC13-007','TC17-032','TC16-020','TC17-035'],
                            'no_double_peak': ['TC11-008','TC09-027','TC17-033','TC17-208','TC17-017','TC15-208','TC16-003']},
               'data'    : [{'control_ts'            : [directory + 'data\\', 'orig_ts']},
                            {'THOR_injury_metrics'   : ['features_TH.csv','stat']},
                            {'H3_injury_metrics'     : ['features_H3.csv','stat']},
                            {'control_injury_metrics': ['features_TH.csv','stat']}],
               'channels': ['11HEAD0000THACXA',
                            '11CHST0000THACXC',
                            '11PELV0000THACXA',
                            '11FEMRLE00THFOZB',
                            '11FEMRRI00THFOZB',
                            '11ACTBLE00THFORB',
                            '11ACTBRI00THFORB',
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
                            '11SEBE0000B3FO0D',
                            '11SEBE0000B6FO0D'],
               'cutoff'  : list(cutoff),
               'test'    : [{'name'        : 'THOR_injury_stats',
                             'test1'       : 'THOR',
                             'test2'       : 'THOR_pair',
                             'paired'      : True,
                             'testname'    : 'wilcoxon',
                             'data'        : 'THOR_injury_metrics',
                             'args'        : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'       : 'H3_injury_stats',
                             'test1'      : 'H3',
                             'test2'      : 'H3_pair',
                             'paired'     : True,
                             'testname'   : 'wilcoxon',
                             'data'       : 'H3_injury_metrics',
                             'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'      : 'control_injury_stats',
                             'test1'     : 'control',
                             'test2'     : 'control_pair',
                             'paired'    : True,
                             'testname'  : 'wilcoxon',
                             'data'      : 'control_injury_metrics',
                             'args'      : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'     : 'chest_double_peak_comparison',
                             'test1'    : 'double_peak',
                             'test2'    : 'no_double_peak',
                             'paired'   : False,
                             'testname' : 'wilcoxon',
                             'data'     : 'control_ts',
                             'args'     : 'exact=FALSE,correct=TRUE,conf.level=0.95'},
                            {'name'        : 'control_stats_ts',
                             'test1'       : 'control',
                             'test2'       : 'control_pair',
                             'paired'      : True,
                             'testname'    : 'wilcoxon',
                             'data'        : 'control_ts',
                             'fwe_correct' : False,
                             'args'        : 'exact=FALSE,correct=TRUE,conf.level=0.95'},], # correction for repeated measures
               'test2'  : None}
    
    with open(directory+'params.json','w') as json_file:
        json.dump(to_JSON,json_file)
