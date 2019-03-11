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
import plotly.graph_objs as go
from plotly.offline import plot
import seaborn as sns
from PMG.COM.easyname import get_units, rename
from PMG.COM.helper import *
from functools import partial
from string import ascii_uppercase as letters

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

names = {'Max_11CHST003STHACRC': 'Chest 3ms clip',
         'Max_11CHST003SH3ACRC': 'Chest 3ms clip',
         'Max_11HEAD003STHACRA': 'Head 3ms clip',
         'Max_11HEAD003SH3ACRA': 'Head 3ms clip',
         'Max_11SEBE0000B6FO0D': 'Lap belt load',
         '11HEAD0000THACXA': 'Head Acx',
         '11FEMRLE00THFOZB': 'Left Femur Axial Load'}
rename = partial(rename, names=names)
#%% time series overlays w/o highlight
plot_channels = ['11HEAD0000THACXA',
                 '11FEMRLE00THFOZB']
subset = table.loc[['TC15-208','TC16-003']]
for ch in plot_channels:
    x = arrange_by_group(subset, chdata[ch], 'Type')
    x['Electric (MY 2016)'] = x.pop('EV')
    x['Conventional (MY 2015)'] = x.pop('ICE')
    fig, ax = plt.subplots()
    ax = plot_overlay(ax, t, x)
    ax = set_labels(ax, {'title': rename(ch), 'xlabel': 'Time [s]', 'ylabel': get_units(ch), 'legend': {'bbox_to_anchor': [1,1]}})
    if ch=='11FEMRLE00THFOZB':
        ax.set_yticks(range(-15000, 10000, 5000))
    ax = adjust_font_sizes(ax, {'title': 24, 'axlabels': 20, 'legend': 20, 'ticklabels': 18})
    fig.savefig(directory + '2019 Presentation\\' + ch + '.png', dpi=600, bbox_inches='tight')
#%% mpl bar plots of individual pairs
plot_channels = ['Max_11CHST003STHACRC',
                 'Max_11HEAD003STHACRA',
                 'Max_11HEAD003SH3ACRA',
                 'Max_11HICR0015THACRA',
                 'Max_11SEBE0000B6FO0D',
                 'Min_11CHST0000THACXC',
                 'Min_11SPIN0100THACXC',
                 'Min_11PELV0000THACXA',
                 'Min_11PELV0000H3ACXA',
                 'Tmin_10SIMELE00INACXD',
                 'Max_11CHST003SH3ACRC',
                 'Min_10SIMELE00INACXD',
                 'Min_11ILACLE00THFOXA',
                 'Min_11ILACRI00THFOXA',
                 'Max_11NECKUP00THFOZA',
                 'Max_13CHST003SHFACRC',
                 'Max_13HEAD0000HFACRA',
                 'Min_13CHST0000HFACXC',
                 'Max_13SEBE0000B6FO0D',
                 'Auc10CVEHCG0000ACXD',
                 'Auc10SIMELE00INACXD',
                 'Auc10SIMERI00INACXD']
plot_channels = ['Auc10CVEHCG0000ACXD',
                 'Auc10SIMELE00INACXD',
                 'Auc10SIMERI00INACXD']

for d in dummies:
    for s in ['Series_1','Series_2']:
        subset = table.query(s + '==1 and ID11==\'{}\''.format(d))
        pairs = subset['Pair_name'].unique()
        codes = letters[:len(pairs)]
        for ch in plot_channels:
            x = intersect_columns(arrange_by_group(subset, features[ch], 'Type', col_key='Pair_name'))
            if x is None or len(x)==0: continue
            x = {k: x[k].abs() for k in x}
            fig, ax = plt.subplots()
            ax = plot_bar(ax, x)
    #        ax.set_xticklabels(list(codes))
            ax = set_labels(ax, {'title': rename(ch) + s, 'ylabel': get_units(ch), 'xlabel': 'Model'})
            ax = adjust_font_sizes(ax, {'title': 24, 'ticklabels': 20, 'axlabels': 20})
            plt.xticks(rotation=90)
            plt.show()
            plt.close(fig)

#%% sns catplot
plot_channels = [['Max_11CHST003STHACRC','Max_11CHST003SH3ACRC'],
                 ['Max_11HEAD003STHACRA','Max_11HEAD003SH3ACRA'],
                 ['Min_11PELV0000THACXA','Min_11PELV0000H3ACXA'],
                 ['Max_11SEBE0000B6FO0D']]


subset = table
pairs = subset.query('Series_1==1 and Type==\'ICE\'')['Pair']
pairs = pairs.append((subset.query('Series_2==1 and Type==\'ICE\'')
                            .apply(lambda x: table.query('Series_2==1 and Model==\'{}\''.format(x['Counterpart'])).index.values[0], axis=1)))

for ch in plot_channels:
    # combine features
    feat_subset = features.loc[subset.index, ch]
    if len(ch)>1:
        feat_subset = feat_subset.dropna(how='all').apply(np.nansum, axis=1).rename('ch').abs()
    else:
        feat_subset = feat_subset.squeeze().rename('ch').abs()
    if len(ch)>1:
        title = ch[0].replace('TH','xx')
    else:
        title = ch[0]
    
    # get differences
    feat_subset = feat_subset.loc[pairs.index] - feat_subset.loc[pairs.values].values
    x = pd.DataFrame({'Model': table.loc[pairs.index, 'Model'], 
                      'Series': ['Series 1 ' + table.at[i, 'ID11'] if table.at[i, 'Series_1']==1 else 'Series 2 TH' for i in pairs.index],
                      'ch': feat_subset})
    ax = sns.boxplot(x='Series', y='ch', data=x, linewidth=1, boxprops={'alpha': 0.5}, width=0.4)
    ax = sns.stripplot(x='Series', y='ch', data=x, color='.25', ax=ax)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
#    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], color=(0.5, 0.5, 0.5), alpha=0.2)
    ax = set_labels(ax, {'title': renameISO(title.lstrip('Min_').lstrip('Max_')), 'xlabel': '', 'ylabel': get_units(title)})
    ax = adjust_font_sizes(ax, {'ticklabels': 14, 'title': 20, 'ylabel': 20})
    plt.savefig(directory + title + '.png', dpi=600, bbox_inches='tight')
    plt.show()

#%% mpl bar plots of aggregated pairs
plot_channels = ['Max_11CHST003STHACRC',
                 'Max_11HEAD003STHACRA',
                 'Max_11HEAD003SH3ACRA',
                 'Max_11HICR0015THACRA',
                 'Max_11SEBE0000B6FO0D',
                 'Min_11CHST0000THACXC',
                 'Min_11SPIN0100THACXC',
                 'Min_11PELV0000THACXA',
                 'Min_11PELV0000H3ACXA',
                 'Tmax_11ACTBLE00THFOXB',
                 'Tmin_10SIMELE00INACXD',
                 'Max_11CHST003SH3ACRC',
                 'Min_10SIMELE00INACXD',
                 'Min_11ILACLE00THFOXA',
                 'Min_11ILACRI00THFOXA']
for t in ['Series_1','Series_2']:
    subset = table.query(t + '==1')
    for ch in plot_channels:
        x = arrange_by_group(subset, features[ch], 'Type')
        if x is None or len(x)==0: continue
        x = {k: x[k].abs().to_frame() for k in x}
        fig, ax = plt.subplots()
        ax = plot_bar(ax, x)
        ax.set_title(' '.join((t, ch)))
        ax.legend()
        plt.show()
        plt.close(fig)
#%% plotly bar plots--individual 
d = 'TH'
ch = 'Max_11CHST003STHACRC'
t = 'Series_2'
def plot_bar_plotly(d, ch, t):
    subset = table.query(t + '==1 and ID11==\'{}\''.format(d))
    x = intersect_columns(arrange_by_group(subset, features[ch], 'Type', col_key='Pair_name'))
    if x is None or len(x)==0: 
        print('data invalid!')
        return
    x = {k: x[k].abs() for k in x}
    indices = get_indices(len(x), x[list(x.keys())[0]].shape[1], 0.6)
    data = []
    for i, k in enumerate(x):
        trace = go.Bar(x = indices[i],
                       y = x[k].mean(),
                       error_y = {'array': x[k].std()},
                       name = k)
        data.append(trace)
        
    layout = go.Layout(showlegend = True,
                       xaxis = go.layout.XAxis(ticktext = x[list(x.keys())[0]].columns,
                                               tickvals = np.mean(indices, axis=0)),
                       title = ' '.join((t, ch)))
    pfig = go.Figure(data=data, layout=layout)
    plot(pfig)
    
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
#    fig.savefig(directory + ch + '.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% regression--difference
# plot: 1. weight vs. response; 2. partner weight vs. response; 3. partner weight/self weight ratio vs. response

ylist = ['Max_11HEAD003STHACRA',
         'Max_11HEAD003SH3ACRA',
         'Max_11HICR0015THACRA',
         'Max_11HICR0015H3ACRA',
         'Max_11CHST003STHACRC',
         'Max_11CHST003SH3ACRC']
ylist = ['Max_11CHST003STHACRC']
xlist = ['Max_13CHST003SHFACRC']

subset = table.query('ID11==\'TH\' and Series_1==1 and Type==\'ICE\'')
for chx in xlist:
    for chy in ylist:
        if chx==chy: continue
        data = pd.DataFrame({'Pair': subset['Pair_name'].values, 
                              'chx': (features.loc[subset.index, chx]-features.loc[subset['Pair'], chx].values),
#                              'chx': features.loc[subset['Pair'], chx].values/features.loc[subset.index, chx].values,
                              'chy': features.loc[subset.index, chy]-features.loc[subset['Pair'], chy].values})
        if data['chx'].isna().all() or data['chy'].isna().all(): continue
        
        fig, ax = plt.subplots()
        ax = sns.scatterplot(x='chx', y='chy', hue='Pair', data=data, ax=ax)
        ax = set_labels(ax, {'xlabel': chx, 'ylabel': chy})
        plt.show()
        plt.close(fig)

#%% regression: no difference
ylist = ['Min_11FEMRLE00THFOZB/partner_weight']
xlist = ['Min_10CVEHCG0000ACXD',
         'Auc_10CVEHCG0000ACXD',
         'Min_10SIMELE00INACXD',
         'Auc_10SIMELE00INACXD']

subset = table.query('ID11==\'TH\' and Series_1==1')
for chx in xlist:
    for chy in ylist:
        if chx==chy: continue
        x = arrange_by_group(subset, features[chx], 'Type')
        y = arrange_by_group(subset, features[chy], 'Type')
        if len(x)==0 or len(y)==0: continue
        match_groups(x,y)
        r2 = [rho(x[k], y[k]) for k in x]
#        if max(r2)<0.3 and min(r2)>-0.3: continue
        
        fig, ax = plt.subplots()
        ax = plot_scatter(ax, x, y)
        ax = set_labels(ax, {'xlabel': chx, 'ylabel': chy, 'legend': {'bbox_to_anchor': (1,1)}})
#        ax.set_title(r2)
        plt.show()
        plt.close(fig)
#%% partner protection with Series 2
#%% mpl bar plots of individual pairs
plot_channels = ['Max_11CHST003STHACRC',
                 'Max_11HEAD003STHACRA',
                 'Max_11HICR0015THACRA',
                 'Max_11SEBE0000B6FO0D',
                 'Min_11CHST0000THACXC',
                 'Min_11SPIN0100THACXC',
                 'Min_11PELV0000THACXA',
                 'Min_10SIMELE00INACXD',
                 'Min_11ILACLE00THFOXA',
                 'Min_11ILACRI00THFOXA',
                 'Max_11NECKUP00THFOZA',
                 'Max_13CHST003SHFACRC',
                 'Max_13HEAD0000HFACRA',
                 'Min_13CHST0000HFACXC',
                 'Min_13SEBE0000B6FO0D']


subset = table.loc[table.query('Series_2==1')['Pair']]
subset['Pair_Type'] = table.loc[subset['Pair'], 'Type'].values
subset['Pair_name'] = table.loc[subset['Pair'], 'Pair_name'].values
for ch in plot_channels:
#    x = intersect_columns(arrange_by_group(subset, features[ch], 'Type', col_key='Pair_name'))
    x = arrange_by_group(subset, features[ch], 'Pair_Type', col_key='Pair_name')
    if x is None or len(x)==0: continue
    x = {k: x[k].abs() for k in x}
    fig, ax = plt.subplots()
    ax = plot_bar(ax, x)
    ax.set_title(ch)
    ax.legend()
    plt.xticks(rotation=90)
    plt.show()
    plt.close(fig)