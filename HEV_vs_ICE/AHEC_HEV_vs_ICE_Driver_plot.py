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
dummy = 'TH'

bounds = pd.read_csv(directory + 'ts_variance.csv',index_col=0)
if dummy=='H3':
    bounds = bounds.rename(mapper=lambda x: x[:10] + 'H3' + x[12:] if x[10:12]=='TH' else x)
lp_abs = bounds['negative']
up_abs = bounds['positive']

plot_channels = [i for i in chdata.columns if (i[10:12]==dummy) or 'VEHCG' in i or 'SEBE' in i or 'SIME' in i]
n = 1
for ch in plot_channels:
    if dummy=='TH':
        fig, axs = plt.subplots(nrows=2,ncols=4,sharey=True,figsize=(15,6))
    else:
        fig, axs = plt.subplots(nrows=2,ncols=3,sharey=True,figsize=(13,6))
    plt.subplots_adjust(hspace=0.3,wspace=0.05)
    i = 0
    for j,tc in enumerate(table.index):
        if j>0 and tc in table['Pair'][:j].values:
                continue
        if not table['Pair'][tc] in chdata.index or not tc in chdata.index:
            if table['Model'][tc] in ['ACCORD','COOPER','ESCAPE','FUSION','JETTA','OPTIMA','PACIFICA','SMART FORTWO','SOUL']:
                i = i + 1
            continue
        x = chdata[ch][tc]
        y = chdata[ch][table['Pair'][tc]]
        
        if len(x) ==1 or len(y)==1:
            continue
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
        ax = mark_diff(ax,tn,xn,yn,lp,up,xlab=table['Model'][tc],ylab=table['Pair_Model'][tc],kernel_size=31,method='diff')
#        ax = mark_diff(ax,tn,xn,yn,-100,100,xlab=table['Model'][tc],ylab=table['Pair Model'][tc],kernel_size=31,method='diff')
        ax.set_title(table['Model'][tc])
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
#%%
if dummy=='TH':
    plot_channels = ['11HEAD0000THACXA','11CHST0000THACXC','11PELV0000THACXA']
else:
    plot_channels = ['11HEAD0000H3ACXA','11CHST0000H3ACXC','11PELV0000H3ACXA']
    
colours = {}
fig,ax = plt.subplots(nrows=1,ncols=3,sharey=True,figsize=(12,3))
for i, ch in enumerate(plot_channels):
    if dummy=='TH':
        ch_ref = ch
    else:
        ch_ref = ch[:10] + 'TH' + ch[12:]
    for j,tc in enumerate(table.index):
        if j>0 and tc in table['Pair'][:j].values:
            continue
        yid = table['Pair'][tc]
        
        
        
        x = tmins[ch + '_Tmin'][yid]-tmins[ch + '_Tmin'][tc]
        y = mins[ch + '_Min'][yid]-mins[ch + '_Min'][tc]
        line = ax[i].plot(x,-y,'o',label=table['Model'][tc])
        if i==0:
            colours[table['Model'][tc]] = line[0].get_markerfacecolor()
    
    k = stat_info.query('channel==\'' + ch_ref + '_Min\' and value==\'lb\' and name==\'control_injury_stats\'').index
    lb = rstats.iloc[0,k].get_values()
    ub = rstats.iloc[0,k+1].get_values()
    k = stat_info.query('channel==\'' + ch_ref + '_Tmin\' and value==\'lb\' and name==\'control_injury_stats\'').index
    tlb = rstats.iloc[0,k].get_values()
    tub = rstats.iloc[0,k+1].get_values()

    
    ax[i].axhline(y=0,color='k',linewidth=1)
    ax[i].axvline(x=0,color='k',linewidth=1)
    ax[i].fill_between(np.concatenate((tlb,tub)),lb,y2=ub,alpha=0.1)
    ax[i].set_title(ch)
    ax[i].set_xlabel('Time Difference [s]')
    if i==0:
        ax[i].set_ylabel('Peak Difference [g]')
    if i==len(plot_channels)-1:
        ax[i].legend(loc=(1.01,0.2))

    print(ch + ':')
    print('lb: ' + str(lb))
    print('ub: ' + str(ub))
    print('tlb: ' + str(tlb))
    print('tub: ' + str(tub))
plt.show()
plt.close(fig)
#%%
if dummy=='TH':
    plot_channels = ['11HICR0015THACRA']    
elif dummy=='H3':
    plot_channels = ['11HICR0015H3ACRA']

if len(plot_channels)%2==0:
    nrow = len(plot_channels)//2
else:
    nrow = len(plot_channels)//2+1


fig,ax = plt.subplots(nrows=nrow,ncols=2,squeeze=False,figsize=(13,2*nrow))
plt.subplots_adjust(hspace=1.5)

for i, ch in enumerate(plot_channels):
    if dummy=='TH':
        ch_ref = ch
    elif dummy=='H3' and not 'DS' in ch:
        ch_ref = ch[:10] + 'TH' + ch[12:]
    else:
        ch_ref = ch[:10] + 'TH' + ch[12:14] + 'R' + ch[15]

    for j,tc in enumerate(table.index):
        if j>0 and table['Pair'][tc] in table.index[:j]:
            continue
        yid = table['Pair'][tc]

        if 'FEMUR' in ch or ('CHST' in ch and 'DSX' in ch):
            y = -(mins[ch + '_Min'][yid]-mins[ch + '_Min'][tc])
        else:
            y = maxs[ch + '_Max'][yid] - maxs[ch + '_Max'][tc]


        ax.flatten()[i].plot(y,0,'|',markersize=15,label=table['Model'][tc],markeredgewidth=2,color=colours[table['Model'][tc]])
    ax.flatten()[i].axhline(y=0,color='k',linewidth=1)
    ax.flatten()[i].axvline(x=0,color='k',linewidth=1)
    
    if 'FEMUR' in ch or ('CHST' in ch and 'DSX' in ch_ref):
        k = stat_info.query('channel==\'' + ch_ref + '_Min\' and value==\'lb\' and name==\'control_injury_stats\'').index
    else:
        k = stat_info.query('channel==\'' + ch_ref + '_Max\' and value==\'lb\' and name==\'control_injury_stats\'').index
    lb = rstats.iloc[0,k].get_values()
    ub = rstats.iloc[0,k+1].get_values()
    
    ax.flatten()[i].fill_between(np.concatenate((lb,ub)),-0.01,y2=0.01,alpha=0.1)
        
    if dummy=='TH' and ('FEMUR' in ch or ('CHST' in ch and 'DSX' in ch)):
        k = stat_info.query('channel==\'' + ch + '_Min\' and value==\'p\' and name==\'THOR_injury_stats\'').index
    elif dummy=='TH':
        k = stat_info.query('channel==\'' + ch + '_Max\' and value==\'p\' and name==\'THOR_injury_stats\'').index
    elif dummy=='H3' and ('FEMUR' in ch or ('CHST' in ch and 'DSX' in ch)):
        k = stat_info.query('channel==\'' + ch + '_Min\' and value==\'p\' and name==\'H3_injury_stats\'').index
    elif dummy=='H3':
        k = stat_info.query('channel==\'' + ch + '_Max\' and value==\'p\' and name==\'H3_injury_stats\'').index

    p = rstats.iloc[0,k][0]

    if p<0.05:
        ax.flatten()[i].set_title(ch.replace('.',' ') + '*')
    else:
        ax.flatten()[i].set_title(ch.replace('.',' '))
    if i in [len(plot_channels)-3, len(plot_channels)-2, len(plot_channels)-1]:
        ax.flatten()[i].set_xlabel('Difference')
    plt.setp((ax.flatten()[i].get_yticklabels() + ax.flatten()[i].get_yticklines() + list(ax.flatten()[i].spines.values())), visible=False)
    ax.flatten()[i].set_ylim(-0.01,0.01)
if not len(plot_channels)%2==0:
    ax.flatten()[-1].set_visible(False)

plt.figure()
for k in colours.keys():
    plt.plot(0,0,'|',markersize=15,label=k,markeredgewidth=2,color=colours[k])
plt.legend(ncol=8,labelspacing=1,loc=(0,1.01))