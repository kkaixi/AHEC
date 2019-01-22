# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:42:40 2018
HEV vs ICE Front Passenger
@author: tangk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMG.read_data import initialize
from PMG.COM.get_props import peakval
from PMG.COM.easyname import renameISO
from PMG.COM.arrange import sep_by_peak, t_ch_from_test_ch

directory = 'C:\\Users\\tangk\\Desktop\\AHEC EV\\'
dummy = 'H3'
cutoff = range(100,1600)
n = 1

channels=['13HEAD0000HFACXA',
          '13CHST0000HFACXC',
          '13PELV0000HFACXA',
          '13FEMRLE00HFFOZB',
          '13FEMRRI00HFFOZB',
          '13SEBE0000B3FO0D',
          '13SEBE0000B6FO0D',
          '13ILACLE00HFFOXA',
          '13ILACRI00HFFOXA']
    
table, t, chdata, se_names = initialize(directory,channels,cutoff,query='ID13==\'HF\'')
#%%
pairs = {'Cruze': ['TC17-025','TC15-035'],
         'Camry': ['TC15-162','TC17-028'],
         'Civic':['TC15-155','TC17-203'],
         'Sentra':['TC16-205','TC17-206'],
         'Explorer':['TC11-233','TC11-234'],
         'Tiguan': ['TC15-029','TC15-030']}

lp_abs = {}
up_abs = {}

for ch in channels:
    for p in pairs.keys():
        xid = pairs[p][0]
        yid = pairs[p][1]
        
        if np.isnan(chdata[ch][xid]).all() or np.isnan(chdata[ch][yid]).all():
            continue
#        plt.plot(t,chdata[ch][xid])
#        plt.plot(t,chdata[ch][yid])
#        plt.title(ch + ' ' + dummy + ' ' + p)
#        plt.show()
        
        dist = get_distribution(chdata[ch][[xid,yid]],0,n=n)
        dist = np.append(dist,get_distribution(chdata[ch][[xid,yid]],1,n=n))
        lp,up = get_pctile(dist)
#        print(str(lp) + ' ' + str(up))
        
        if not ch in lp_abs.keys():
            lp_abs[ch] = lp
            up_abs[ch] = up
        else:
            if lp < lp_abs[ch]:
                lp_abs[ch] = lp
            if up > up_abs[ch]:
                up_abs[ch] = up

#%% visualize pairs
for ch in channels:
    fig, axs = plt.subplots(nrows=5,ncols=3,sharey=True,figsize=(15,25))
    i = 0
    for j,tc in enumerate(table.query('Subset==\'HEV vs ICE\'').index):
        if j>0:
            if tc in table['Pair'][:j].values:
                continue
        if not table['Pair'][tc] in chdata.index or not tc in chdata.index:
            continue
        x = chdata[ch][tc]
        y = chdata[ch][table['Pair'][tc]]
        
#        ax = plt.axes()
        ax = axs.flatten()[i]
        ax.plot(t,x,color='b',label=table['Model'][tc])
        ax.plot(t,y,color='k',label=table['Pair Model'][tc])
        ax.set_title(table['Model'][tc])
        ax.legend()
        i = i + 1
    fig.suptitle(ch)
    fig.savefig(directory + 'Passenger\\' + ch + '.png',bbox_inches='tight')