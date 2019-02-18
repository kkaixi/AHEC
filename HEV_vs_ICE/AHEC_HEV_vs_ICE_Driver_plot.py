# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:09:11 2018
Plots for HEV vs ICE Driver analysis 
@author: tangk
"""
import matplotlib.pyplot as plt
import pandas as pd
from PMG.COM.timeseries import mark_diff
import json
from PMG.read_data import read_table
from AHEC_HEV_vs_ICE_Driver_initialize import *
from PMG.COM.arrange import *
from PMG.COM.plotfuns import *
from PMG.read_data import reread_table

directory = 'P:\\Data Analysis\\Projects\\AHEC EV\\'
with open(directory+'params.json','r') as json_file:
    params = json.load(json_file)

res = {}
for data in params['test']:
    label1 = data['label1']
    label2 = data['label2']
    index = pd.MultiIndex.from_arrays([label1,label2])
    res[data['name'][0]] = pd.Series(data['res'],index=index).unstack().dropna(axis=0, how='all').astype(np.float32)

dummies = ['TH','H3']
#%%
plot_channels = ['Max_11CHST003STHACRC',
                 'Max_11HEAD003STHACRA',
                 'Max_11HICR0015THACRA',
                 'Max_11SEBE0000B6FO0D',
                 'Min_11CHST0000THACXC',
                 'Min_11SPIN0100THACXC',
                 'Tmax_11ACTBLE00THFOXB',
                 'Tmin_10SIMELE00INACXD',
                 'Max_11CHST003SH3ACRC',
                 'Min_10SIMELE00INACXD',
                 'Min_11ILACLE00THFOXA',
                 'Min_11ILACRI00THFOXA']

for t in ['Series_1','Series_2']:
    subset = table.query(t + '==1')
    for ch in plot_channels:
        x = intersect_columns(arrange_by_group(subset, features[ch], 'Type', col_key='Pair_name'))
        if x is None or len(x)==0: continue
        fig, ax = plt.subplots()
        ax = plot_bar(ax, x)
        ax.set_title(' '.join((t, ch)))
        ax.legend()
        plt.xticks(rotation=90)


#%% plot pairs using CIs from probability distribution of difference
        
subset = table.query('ID11==\'H3\' and Series_1==1 and Type==\'ICE\'')
bounds = pd.read_csv(directory + 'ts_variance.csv',index_col=0)['negative']
plot_channels = [i for i in chdata.columns if (i[10:12]=='H3') or 'VEHCG' in i or 'SEBE' in i or 'SIME' in i]

for ch in plot_channels:
    if chdata.loc[subset.index, ch].apply(is_all_nan).all(): continue
    fig, axs = get_axes(len(subset))
    for i, tc in enumerate(subset.index):
        if tc not in chdata.index: continue
        x = chdata.at[tc, ch]
        pair = subset.at[tc, 'Pair']
        if pair not in chdata.index: continue
        y = chdata.at[pair, ch]
        bd = abs(bounds[ch]) if ch in bounds.index else 1000
        ax = axs.flatten()[i]
        ax = mark_diff(ax, t, x, y, -bd, bd, xlab=subset.at[tc,'Model'], ylab=subset.at[tc, 'Pair_Model'], kernel_size=31, method='diff')
        ax.set_title(subset.at[tc,'Model'])
        ax.legend()
    plt.suptitle(ch)
    fig.savefig(directory + ch + '.png', bbox_inches='tight')
#    plt.show()
    plt.close(fig)