# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:35:25 2018

@author: tangk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMG.read_data import initialize
from PMG.COM.get_props import peakval, get_ipeak
from PMG.COM.easyname import renameISO, rename_list
from PMG.COM.arrange import sep_by_peak, t_ch_from_test_ch
import json


directory = 'C:\\Users\\tangk\\Desktop\\AHEC Old vs New\\'
cutoff = range(100,1600)

channels = ['10CVEHCG0000ACXD',
            '10CVEHCG0000ACRD',
            '11HEAD0000THACXA',
            '11HEAD0000THACYA',
            '11HEAD0000THACZA',
            '11HEAD0000THACRA',
            '11CHST0000THACXC',
            '11PELV0000THACXA',
            '11HICR0015THACRA',
            '11NIJCIPTETH00YX',
            '11NIJCIPTFTH00YX',
            '11NIJCIPCFTH00YX',
            '11NIJCIPCETH00YX',
            '11FEMRLE00THFOZB',
            '11FEMRRI00THFOZB',
            '11CHSTRIUPTHDSRB',
            '11CHSTRILOTHDSRB',
            '11CHSTLEUPTHDSRB',
            '11CHSTLELOTHDSRB',
            '11HEAD0000H3ACXA',
            '11HEAD0000H3ACYA',
            '11HEAD0000H3ACZA',
            '11HEAD0000H3ACRA',
            '11CHST0000H3ACXC',
            '11CHST0000H3ACRC',
            '11PELV0000H3ACXA',
            '11PELV0000H3ACZA',
            '11PELV0000H3ACRA',
            '11HICR0015H3ACRA',
            '11NIJCIPTEH300YX',
            '11NIJCIPTFH300YX',
            '11NIJCIPCFH300YX',
            '11NIJCIPCEH300YX',
            '11CHST0000H3DSXB',
            '11FEMRLE00H3FOZB',
            '11FEMRRI00H3FOZB',
            '10SIMELE00INACXD',
            '10SIMERI00INACXD']
drop = []

table, t, chdata, se_names = initialize(directory,channels,cutoff)
table = table.drop(drop)

preprocess = ['11CHSTLEUPTHDSRB','11CHSTRIUPTHDSRB','11CHSTLELOTHDSRB','11CHSTRILOTHDSRB']
chdata[preprocess] = chdata[preprocess].applymap(lambda x: x-x[0])
chdata['11CHST0000THDSRB'] = chdata[preprocess].applymap(max).max(axis=1).apply(lambda x: [x])
#chdata = chdata.drop(preprocess,axis=1)

#%%
if __name__=='__main__':
    def i_to_t(i):
        if not np.isnan(i):
            return t[int(i)]
        else:
            return np.nan
    
    mins, maxs = sep_by_peak(chdata.applymap(peakval))
    tmins,tmaxs = sep_by_peak(chdata.applymap(get_ipeak))
    mins = mins.rename(lambda name:(name+'_Min'),axis=1)
    maxs = maxs.rename(lambda name:(name+'_Max'),axis=1)
    tmins = tmins.rename(lambda name:(name+'_Tmin'),axis=1).applymap(i_to_t)
    
    #%%
    for dummy in ['TH','H3']:
        ch = [i for i in chdata.columns if i[10:12]==dummy]
        ch_mins = [i +'_Min' for i in ch if (i[2:6] in ['HEAD','CHST','PELV','FEMR'] and not i[14]=='R')]
        ch_maxs = [i +'_Max' for i in ch if not (i[2:] in ['HEAD','CHST','PELV','FEMR'] and not i[14]=='R')]
        ch_tmins = [i + '_Tmin' for i in ch if (i[2:6] in ['HEAD','CHST','PELV','FEMR'] and not i[14]=='R')]
        
        stattable = pd.concat((mins[ch_mins],maxs[ch_maxs],tmins[ch_tmins]),axis=1).rename_axis('TC')
        stattable.to_csv(directory + 'stattable_' + dummy + '.csv')
    
    #%%
    to_JSON = {'project_name': 'OLD_vs_NEW_Driver',
               'directory'   : directory,
               'cat'     : {'THOR'        : list(table.query('ID11==\'TH\' and Subset==\'OLD vs NEW\'').index[0::2]),
                            'THOR_pair'   : list(table.query('ID11==\'TH\' and Subset==\'HEV vs ICE\'').index[1::2]),
                            'H3'          : list(table.query('ID11==\'H3\' and Subset==\'HEV vs ICE\'').index[0::2]),
                            'H3_pair'     : list(table.query('ID11==\'H3\' and Subset==\'HEV vs ICE\'').index[1::2])},
               'data'    : [{'THOR_injury_metrics'   : ['stattable_TH.csv','stat']},
                            {'H3_injury_metrics'     : ['stattable_H3.csv','stat']}],
               'channels': ['11HEAD0000THACXA',
                            '11CHST0000THACXC',
                            '11PELV0000THACXA',
                            '11FEMRLE00THFOZB',
                            '11FEMRRI00THFOZB'],
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
                              'args'       : 'exact=FALSE,correct=TRUE,conf.level=0.95'}], # correction for repeated measures
               'test2'  : None}
    
    with open(directory+'params.json','w') as json_file:
        json.dump(to_JSON,json_file)
