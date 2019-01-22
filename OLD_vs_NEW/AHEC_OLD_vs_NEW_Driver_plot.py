# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:02:41 2018

@author: tangk
"""

import matplotlib.pyplot as plt
import pandas as pd
from PMG.COM.timeseries import mark_diff
import json
from PMG.read_data import read_table
import numpy as np
from AHEC_OLD_vs_NEW_Driver import *

dummy = 'H3'
plot = 1
savefig = 0

directory = 'C:\\Users\\tangk\\Desktop\\AHEC Old vs New\\'
rstats = pd.read_csv('C:\\Users\\tangk\\Desktop\\AHEC EV\\Rstats.csv',index_col=0,dtype=np.float64)
with open(directory+'params.json','r') as json_file:
    to_JSON = json.load(json_file)
with open('C:\\Users\\tangk\\Desktop\\AHEC EV\\params.json','r') as json_file:
    to_JSON2 = json.load(json_file)
    

stat_info = pd.DataFrame(to_JSON2['stats_label'])
table = read_table(directory + 'Table.csv').query('ID11==\'' + dummy + '\'')

#%%

bounds = pd.read_csv('C:\\Users\\tangk\\Desktop\\AHEC EV\\ts_variance.csv',index_col=0)
if dummy=='H3':
    bounds.at['11CHST0000THDSXB','negative'] = bounds.loc[['11CHSTLEUPTHDSXB','11CHSTRIUPTHDSXB','11CHSTLELOTHDSXB','11CHSTRILOTHDSXB']]['negative'].min()
    bounds.at['11CHST0000THDSXB','positive'] = bounds.loc[['11CHSTLEUPTHDSXB','11CHSTRIUPTHDSXB','11CHSTLELOTHDSXB','11CHSTRILOTHDSXB']]['positive'].max()
    bounds = bounds.rename(mapper=lambda x: x[:10] + 'H3' + x[12:] if x[10:12]=='TH' else x)
lp_abs = bounds['negative']
up_abs = bounds['positive']


plot_channels = [i for i in chdata.columns if (i[10:12]==dummy and i[2:6] in ['CHST','FEMR','HEAD','PELV']) or ('VEHCG' in i) or ('SIME' in i)]
n = 1
for ch in plot_channels:

#    if dummy=='TH':
    fig, axs = plt.subplots(nrows=2,ncols=4,sharey=True,figsize=(15,6))
#    else:
#        fig, axs = plt.subplots(nrows=2,ncols=3,sharey=True,figsize=(13,6))
    plt.subplots_adjust(hspace=0.3,wspace=0.05)
    i = 0
    for j,tc in enumerate(table.index):
        if j>0 and tc in table['Pair'][:j].values:
                continue
        if not table['Pair'][tc] in chdata.index or not tc in chdata.index:
            continue
        x = chdata[ch][tc]
        y = chdata[ch][table['Pair'][tc]]
        xn = x[0::n]
        yn = y[0::n]
        tn = t[0::n]
        
        if not ch in lp_abs:
            lp = -100
            up = 100
        else:
            lp = lp_abs[ch]
            up = up_abs[ch]
        
        
#        ax = plt.axes()
        ax = axs.flatten()[i]
        ax = mark_diff(ax,tn,xn,yn,lp,up,xlab=table['Model'][tc],ylab=table['Pair Model'][tc],kernel_size=31,method='diff')
        ax.set_title(table['Model'][tc][:-3])
        ax.legend(fontsize=8)
        if dummy=='TH':
            if i==0 or i==4:
                ax.set_ylabel('Acceleration [g]')
            if i>=4:
                ax.set_xlabel('Time [s]')
        if dummy=='H3':
            if i==0 or i==3:
                ax.set_ylabel('Acceleration [g]')
            if i>=3:
                ax.set_xlabel('Time [s]')
        i = i + 1

    fig.suptitle(ch)
    if plot:
        plt.show()
    if savefig:
        fig.savefig(directory + ch + '_n=' + str(n) + '.png',bbox_inches='tight')
    plt.close(fig)