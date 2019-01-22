# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:44:15 2018
function to test applicability of paired comparison 
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
from PMG.COM.timeseries import get_distribution
import seaborn
from scipy.stats import mannwhitneyu as mwu
import random

directory = 'C:\\Users\\tangk\\Desktop\\AHEC EV\\'
cutoff = range(100,1600)

channels = ['10SIMELE00INACXD',
            '10SIMERI00INACXD']

#%%
table, t, chdata, se_names = initialize(directory,channels,cutoff,query='Subset==\'HEV vs ICE\' and ID11==\'TH\'')
chdata['LE-RI'] = chdata['10SIMELE00INACXD'] - chdata['10SIMERI00INACXD']
channels.append('LE-RI')

for ch in channels:

    fig, axs = plt.subplots(nrows=2,ncols=4,sharey=True,figsize=(15,6))
    plt.subplots_adjust(hspace=0.3,wspace=0.05)
    i = 0
    
    for j,tc in enumerate(table.index):
        if table['Model'][tc] in ['CRUZE','ACCORD']:
            continue
        if j>0 and tc in table['Pair'][:j].values:
                continue
        if not table['Pair'][tc] in chdata.index or not tc in chdata.index:
            if table['Model'][tc] in ['ACCORD','COOPER','ESCAPE','FUSION','JETTA','OPTIMA','PACIFICA','SMART FORTWO','SOUL']:
                i = i + 1
            continue
        x = chdata[ch][tc]
        y = chdata[ch][table['Pair'][tc]]

        
        ax = axs.flatten()[i]
        ax.plot(t,x,label=table['Model'][tc],color='b')
        ax.plot(t,y,label=table['Pair Model'][tc],color='k')
        ax.set_title(table['Model'][tc])
        ax.legend(fontsize=8)

        if i==0 or i==4:
            ax.set_ylabel('Acceleration [g]')
        if i>=4:
            ax.set_xlabel('Time [s]')

        i = i + 1

    fig.suptitle(ch)
    plt.show()


#%%
table, t, chdata, se_names = initialize(directory,channels,cutoff,query='(Model==\'SOUL\' | Model==\'SOUL EV\') & ID11==\'TH\'')

for ch in channels:
    fig, axs = plt.subplots(nrows=1,ncols=3,sharey=True,figsize=(15,4))
    j = 0
    for i in table.index:
        if i=='TC16-003':
            continue
        axs[j].plot(t,chdata[ch]['TC16-003'],color='k',label='Soul EV')
        axs[j].plot(t,chdata[ch][i],color='b',label=i)
        axs[j].legend()
        axs[j].set_title(ch)
        j = j + 1
