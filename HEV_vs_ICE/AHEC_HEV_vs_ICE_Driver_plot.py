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
import re

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

names = {'Min_11HEAD0000THACXA': 'Head Acx',
         'Min_11HEAD0000H3ACXA': 'Head Acx',
         'Min_13HEAD0000HFACXA': 'Head Acx',
         'Min_11HEAD0000xxACXA': 'Head Acx',
         'Max_11HEAD0000xxACRA': 'Head AcR',
         'Max_11HEAD0000THACRA': 'Head AcR',
         'Max_13HEAD0000HFACRA': 'Head AcR',
         'Min_11CHST0000THACXC': 'Chest Acx',
         'Min_11CHST0000H3ACXC': 'Chest Acx',
         'Min_13CHST0000HFACXC': 'Chest Acx',
         'Min_11CHST0000xxACXC': 'Chest Acx',
         'Max_11CHST0000xxACRC': 'Chest AcR',
         'Max_11CHST0000THACRC': 'Chest AcR',
         'Max_13CHST0000HFACRC': 'Chest AcR',
         'Min_11SPIN0100THACXC': 'Upper Spine Acx',
         '11HEAD0000THACXA': 'Head Acx',
         '11FEMRLE00THFOZB': 'Left Femur Fz',
         '11FEMRRI00THFOZB': 'Right Femur Fz',

         'Max_11NECKUP00THFOZA': 'Upper Neck Fz',
         'Max_11NECKUP00H3FOZA': 'Upper Neck Fz',
         'Max_13NECKUP00HFFOZA': 'Upper Neck Fz',
         'Max_11NECKUP00xxFOZA': 'Upper Neck Fz',
         'TH': 'THOR',
         'H3': 'Hybrid III',
         'HF': 'Hybrid III Female',
         'Min_11PELV0000xxACXA': 'Pelvis Acx',
         'Min_13PELV0000HFACXA': 'Pelvis Acx',
         'Min_10CVEHCG0000ACXD': 'Vehicle CG Acx',
         'Min_10SIMELE00INACXD': 'Left B-Pillar Acx',
         'OLC_10SIMELE00INACXD': 'OLC (Left B-Pillar)',
         'OLC_10SIMERI00INACXD': 'OLC (Right B-Pillar)'}
 
rename = partial(rename, names=names)
#%% time series overlays w/o highlight
plot_channels = ['11HEAD0000THACXA',
                 '11HEAD0000THACRA',
                 '11FEMRLE00THFOZB',
                 '11FEMRRI00THFOZB']
grouped = table.table.query_list('Model',['SOUL','SOUL EV']).groupby(['Pair_name','Speed'])
for i, grp in enumerate(grouped):
    subset = grp[1]
    subset['TC'] = subset.index
    for ch in plot_channels:
        x = arrange_by_group(subset, chdata[ch], 'Type')
#        x['Electric (MY 2016)'] = x.pop('EV')
#        x['Conventional (MY 2015)'] = x.pop('ICE')
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, t, x, line_specs={'EV': {'color': (34/255, 89/255, 149/255)}, 'ICE': {'color': (205/255, 0, 26/255)}})
        ax = set_labels(ax, {'title': rename(ch), 'xlabel': 'Time [s]', 'ylabel': get_units(ch)})
        ax = adjust_font_sizes(ax, {'title': 24, 'axlabels': 20, 'ticklabels': 18})
#        ax = set_labels(ax, {'title': '{0} (Test {1})'.format(rename(ch), 2-i), 'xlabel': 'Time [s]', 'ylabel': get_units(ch), 'legend': {'bbox_to_anchor': [1,1]}})
#        ax.legend(ncol=2)
#        ax = adjust_font_sizes(ax, {'title': 24, 'axlabels': 20, 'legend': 20, 'ticklabels': 18})
        if 'FOZ' in ch:
            ax.set_ylabel(ax.get_ylabel() + r' ($\times 10^4$)')
            ax.set_ylim(-14000, 3800)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,1))
        elif 'ACX' in ch:
            ax.set_ylim(-70, 10)
        plt.show()
        plt.close(fig)
#%% sns catplot
plot_channels = [['Max_11HEAD0000THACRA','Max_11HEAD0000H3ACRA'],
                 ['Min_11HEAD0000THACXA','Min_11HEAD0000H3ACXA'],
                 ['Max_11NECKUP00THFOZA','Max_11NECKUP00H3FOZA'],
                 ['Max_11CHST0000THACRC','Max_11CHST0000H3ACRC'],
                 ['Min_11CHST0000THACXC','Min_11CHST0000H3ACXC'],
                 ['Min_11PELV0000THACXA','Min_11PELV0000H3ACXA']]

subset = table.query('ID13!=\'YA\'')

pairs = subset.query('Series_1==1 and Speed==48 and Type==\'ICE\'')[['Pair','Pair_name','ID11']]
pairs['Series'] = ''

s2 = subset.query('Series_2==1 and Speed==48 and Type==\'ICE\'')
ev_s2 = s2.apply(lambda x: table.query('Series_2==1 and Model==\'{}\''.format(x['Counterpart'])).index.values[0], axis=1)
s2 = s2[['Pair_name','ID11']]
s2['Pair'] = ev_s2
s2['Series'] = '\n(Series 2)'
pairs = pairs.append(s2)
pairs['ID11'] = pairs['ID11'].apply(rename)
pairs['Label'] = pairs[['ID11','Series']].apply(lambda x: ''.join(x), axis=1)

r = re.compile('_\d\d')

pvals = pd.DataFrame({'THOR': res['Series_1_48_THOR_t'].loc[np.concatenate(plot_channels), 'p'],
                      'Hybrid III': res['Series_1_48_H3_t'].loc[np.concatenate(plot_channels), 'p'],
                      'THOR\n(Series 2)': res['Series_2_48_t'].loc[np.concatenate(plot_channels), 'p']})
order = ['THOR\n(Series 2)', 'Hybrid III', 'THOR']
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
    id_number = r.search(title).group()[1:]
    
    # get differences
    feat_subset = feat_subset.loc[pairs.index] - feat_subset.loc[pairs['Pair']].values
    pairs['ch'] = feat_subset

#    ax = sns.boxplot(x='Label', y='ch', data=pairs, linewidth=1, 
#                     boxprops={'alpha': 0.5}, width=0.4, sym='')
    fig, ax = plt.subplots()
    sns.despine(ax=ax, bottom=True, top=True, left=True)
    ax = sns.pointplot(y='Label', x='ch', order=order, data=pairs, join=False, ci='sd', err_style='bars', capsize=0.2, color='.25', errwidth=2, orient='h', ax=ax)
    ax = sns.stripplot(y='Label', x='ch', order=order, data=pairs, color='.25', ax=ax, orient='h')
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
#    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], color=(0.5, 0.5, 0.5), alpha=0.2)
    ax = set_labels(ax, {'title': rename(title), 'ylabel': '', 'xlabel': 'Difference in ' + get_units(title)})
    ax = adjust_font_sizes(ax, {'ticklabels': 20, 'title': 24, 'xlabel': 20})
    ax.set_xlim([i*2 for i in ax.get_xlim()])
    ax.set_ylim([-0.5, 3.5])
    add_stars(ax, 
#              1.7*pairs.groupby('Label').max().loc[order, 'ch'].values, 
              [0.8*ax.get_xlim()[1]]*3,
              pvals.loc[ch].apply(np.nansum)[order].values, 
              ax.get_yticks(), 
              orientation='h',
              fontsize=24)
    plt.locator_params(axis='x', nbins=7)
#    plt.savefig(directory + title + '.png', dpi=600, bbox_inches='tight')
    plt.show()
    print(ch)
    print(pairs.groupby('Label').mean())
    

#%% sns catplot HF
plot_channels = [['Max_13HEAD0000HFACRA'],
                 ['Min_13HEAD0000HFACXA'],
                 ['Max_13NECKUP00HFFOZA'],
                 ['Max_13CHST0000HFACRC'],
                 ['Min_13CHST0000HFACXC'],
                 ['Min_13PELV0000HFACXA']]

subset = table.query('ID13!=\'YA\'')

pairs = subset.query('Series_1==1 and Speed==48 and Type==\'ICE\'')[['Pair','Pair_name','ID13']]
pairs['Series'] = '(Series 1)'

s2 = subset.query('Series_2==1 and Speed==48 and Type==\'ICE\'')
ev_s2 = s2.apply(lambda x: table.query('Series_2==1 and Model==\'{}\''.format(x['Counterpart'])).index.values[0], axis=1)
s2 = s2[['Pair_name','ID13']]
s2['Pair'] = ev_s2
s2['Series'] = '(Series 2)'
pairs = pairs.append(s2)
pairs['ID13'] = pairs['ID13'].apply(rename)
pairs['Label'] = pairs[['ID13','Series']].apply(lambda x: '\n'.join(x), axis=1)

r = re.compile('_\d\d')

order = ['Hybrid III Female\n(Series 2)', 'Hybrid III Female\n(Series 1)']

pvals = pd.DataFrame({'Hybrid III Female\n(Series 2)': res['Series_2_48_t'].loc[np.concatenate(plot_channels), 'p'],
                       'Hybrid III Female\n(Series 1)': res['Series_1_48_HF_t'].loc[np.concatenate(plot_channels), 'p']})
    
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
    id_number = r.search(title).group()[1:]
    
    # get differences
    feat_subset = feat_subset.loc[pairs.index] - feat_subset.loc[pairs['Pair']].values
    pairs['ch'] = feat_subset
    
    fig, ax = plt.subplots()
    sns.despine(ax=ax, bottom=True, top=True, left=True)
    ax = sns.pointplot(y='Label', x='ch', order=order, data=pairs, join=False, ci='sd', err_style='bars', capsize=0.2, color='.25', errwidth=2, orient='h', ax=ax)
    ax = sns.stripplot(y='Label', x='ch', order=order, data=pairs, color='.25', ax=ax, orient='h')
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
#    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], color=(0.5, 0.5, 0.5), alpha=0.2)
    ax = set_labels(ax, {'title': rename(title), 'ylabel': '', 'xlabel': 'Difference in ' + get_units(title)})
    ax = adjust_font_sizes(ax, {'ticklabels': 20, 'title': 24, 'xlabel': 20})
    ax.set_xlim([i*2 for i in ax.get_xlim()])
    ax.set_ylim([-0.5, 1.5])
    add_stars(ax, 
#              1.7*pairs.groupby('Label').max().loc[order, 'ch'].values, 
              [0.8*ax.get_xlim()[1]]*2,
              pvals.loc[ch, order].values[0], 
              ax.get_yticks(), 
              orientation='h',
              fontsize=24)
    plt.locator_params(axis='x', nbins=8)
#    plt.savefig(directory + title + '.png', dpi=600, bbox_inches='tight')
    plt.show()
    print(ch)
    print(pairs.groupby('Label').mean())

#%% sns catplot for vehicle stuff
plot_channels = [['Min_10CVEHCG0000ACXD'],
                 ['Min_10SIMELE00INACXD'],
                 ['OLC_10SIMELE00INACXD'],
                 ['OLC_10SIMERI00INACXD']]

#subset = table.query('ID13!=\'YA\'')
subset = table.loc[table['13'].query('ID13!=\'YA\'').index]

pairs = subset.loc[subset['10'].query('Series_1==1 and Speed==48 and Type==\'ICE\'').index][[('10','Pair'),('10','Pair_name'),('11','ID11')]]
pairs.columns = pairs.columns.get_level_values(1)
pairs['Series'] = 'Series 1'

s2 = subset.loc[subset['10'].query('Series_2==1 and Speed==48 and Type==\'ICE\'').index]
s2.columns = s2.columns.get_level_values(1)
ev_s2 = s2.apply(lambda x: table['10'].query('Series_2==1 and Model==\'{}\''.format(x['Counterpart'])).index.values[0], axis=1)
s2 = s2[['Pair_name','ID11']]
s2['Pair'] = ev_s2
s2['Series'] = 'Series 2'
pairs = pairs.append(s2)

r = re.compile('_\d\d')

order = ['Series 2','Series 1']
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
    feat_subset = feat_subset.loc[pairs.index] - feat_subset.loc[pairs['Pair']].values
    pairs['ch'] = feat_subset

#    ax = sns.boxplot(x='Label', y='ch', data=pairs, linewidth=1, 
#                     boxprops={'alpha': 0.5}, width=0.4, sym='')
    fig, ax = plt.subplots()
    sns.despine(ax=ax, bottom=True, top=True, left=True)
    ax = sns.pointplot(y='Series', x='ch', order=order, data=pairs, join=False, ci='sd', err_style='bars', capsize=0.2, color='.25', errwidth=2, orient='h', ax=ax)
    ax = sns.stripplot(y='Series', x='ch', order=order, data=pairs, color='.25', ax=ax, orient='h')
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
#    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], color=(0.5, 0.5, 0.5), alpha=0.2)
    ax = set_labels(ax, {'title': rename(title), 'ylabel': '', 'xlabel': 'Difference in ' + get_units(title)})
    ax = adjust_font_sizes(ax, {'ticklabels': 20, 'title': 24, 'xlabel': 20})
    ax.set_xlim([i*2 for i in ax.get_xlim()])
    ax.set_ylim([-0.5, 2.5])
    plt.locator_params(axis='x', nbins=7)
#    plt.savefig(directory + title + '.png', dpi=600, bbox_inches='tight')
    plt.show()
    print(ch)
    print(pairs.groupby('Series').mean())
    

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


#%% create a linear regression model and use it to predict responses based on weight/partner weight to 
# assess how much the two account for observed differences in the response
to_json_2 = {}
from sklearn.linear_model import LinearRegression
ylist = ['Max_11HEAD0000THACRA',
#         'Max_11NECKUP00THFOZA',
         'Max_11CHST0000THACRC',
         'Min_11CHST0000THACXC',
         'Max_13HEAD0000HFACRA',
         'Min_13HEAD0000HFACXA',
         'Max_13CHST0000HFACRC',
         'Min_13CHST0000HFACXC']
xlist = [['Weight']]



subset = table.query('Type==\'ICE\' and Speed==48')
subset_test = table.query('Type==\'EV\' and Speed==48')
for chx in xlist:
    to_json_2['predict_ev'] = {}
    error = []
    for chy in ylist:
        x = features.loc[subset.index, chx]
        y = features.loc[subset.index, chy].abs()
        i = ~(x.isna().any(axis=1) | y.isna())
        i = i[i].index
        x, y = x.loc[i].values, y[i].to_frame().values
        
        x_test = features.loc[subset_test.index, chx]
        y_test = features.loc[subset_test.index, chy].abs()
        i_test = ~(x_test.isna().any(axis=1) | y_test.isna())
        i_test = i_test[i_test].index
        x_test, y_test = x_test.loc[i_test].values, y_test[i_test].to_frame().values
        y_test = np.squeeze(y_test)
        
        lr = LinearRegression()
        lr = lr.fit(x, y)
        rsq = lr.score(x, y)
        print(chy, chx, rsq)
        y_pred = np.squeeze(lr.predict(x_test))
        err = y_pred-y_test
        to_json_2['predict_ev'][chy] = list(err)
        response_name = 'THOR ' + rename(chy) if '11' in chy else 'HF ' + rename(chy)
        error.append(pd.DataFrame({'Value': np.concatenate((y_pred, y_test, np.squeeze(y))),
                                   'Response_type': np.concatenate((np.tile('Predicted EV',len(y_pred)), 
                                                              np.tile('Actual EV',len(y_test)),
                                                              np.tile('ICE', len(y)))),
                                   'Response_name': '\n'.join(response_name.split())}))
    error = pd.concat(error)
    fig, ax = plt.subplots(figsize=(8,4))
    ax = sns.pointplot(x='Response_name', 
                       y='Value', 
                       hue='Response_type', 
                       hue_order=['ICE','Predicted EV','Actual EV'], 
                       palette={'ICE': (205/255, 0, 26/255), 
                                'Predicted EV': (116/255, 36/255, 191/255), 
                                'Actual EV':(34/255, 89/255, 149/255)},
                       data=error, 
                       join=False, 
                       ci='sd', 
                       err_style='bars', 
                       capsize=0.2, 
                       errwidth=2, 
                       dodge=0.45, 
                       ax=ax)
    
    ax = sns.stripplot(x='Response_name', 
                       y='Value', 
                       hue='Response_type', 
                       hue_order=['ICE','Predicted EV','Actual EV'], 
                       palette={'ICE': (205/255, 0, 26/255), 
                                'Predicted EV': (116/255, 36/255, 191/255), 
                                'Actual EV':(34/255, 89/255, 149/255)},
                       alpha=0.3, 
                       data=error, 
                       dodge=0.45)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:], labels[3:], bbox_to_anchor=(0.9,-0.4), ncol=3)
    ax = set_labels(ax, {'title': 'Predicted EV Response', 'ylabel': get_units(chy[5:]), 'xlabel': ''})
    ax = adjust_font_sizes(ax, {'ticklabels': 16, 'title': 20, 'axlabels': 20, 'legend': 18})
#    ax.tick_params(axis='x', rotation=90)

#%% same as above but backwards
from sklearn.linear_model import LinearRegression
ylist = ['Max_11HEAD0000THACRA',
#         'Max_11NECKUP00THFOZA',
         'Max_11CHST0000THACRC',
         'Min_11CHST0000THACXC',
         'Max_13HEAD0000HFACRA',
         'Min_13HEAD0000HFACXA',
         'Max_13CHST0000HFACRC',
         'Min_13CHST0000HFACXC']
xlist = [['Weight']]



subset = table.query('Type==\'ICE\' and Speed==48')
subset_test = table.query('Type==\'EV\' and Speed==48')
for chx in xlist:
    error = []
    to_json_2['predict_ice'] = {}
    for chy in ylist:
        x = features.loc[subset.index, chx]
        y = features.loc[subset.index, chy].abs()
        i = ~(x.isna().any(axis=1) | y.isna())
        i = i[i].index
        x, y = x.loc[i].values, y[i].to_frame().values
        y = np.squeeze(y)
        
        x_test = features.loc[subset_test.index, chx]
        y_test = features.loc[subset_test.index, chy].abs()
        i_test = ~(x_test.isna().any(axis=1) | y_test.isna())
        i_test = i_test[i_test].index
        x_test, y_test = x_test.loc[i_test].values, y_test[i_test].to_frame().values
        y_test = np.squeeze(y_test)
        
        lr = LinearRegression()
        lr = lr.fit(x_test, y_test)
        rsq = lr.score(x_test, y_test)
        print(chy, chx, rsq)
        y_pred = np.squeeze(lr.predict(x))
        err = y_pred-y
        to_json_2['predict_ice'][chy] = list(err)
        response_name = 'THOR ' + rename(chy) if '11' in chy else 'HF ' + rename(chy)
        error.append(pd.DataFrame({'Value': np.concatenate((y_pred, y, np.squeeze(y_test))),
                                   'Response_type': np.concatenate((np.tile('Predicted ICE',len(y_pred)), 
                                                              np.tile('Actual ICE',len(y)),
                                                              np.tile('EV', len(y_test)))),
                                   'Response_name': '\n'.join(response_name.split())}))
    error = pd.concat(error)
    fig, ax = plt.subplots(figsize=(8,4))
    ax = sns.pointplot(x='Response_name', 
                       y='Value', 
                       hue='Response_type', 
                       hue_order=['EV','Predicted ICE','Actual ICE'],  
                       palette={'Actual ICE': (205/255, 0, 26/255), 
                                'Predicted ICE': (186/255, 51/255, 168/255), 
                                'EV':(34/255, 89/255, 149/255)},
                       data=error, 
                       join=False, 
                       ci='sd', 
                       err_style='bars', 
                       capsize=0.2, 
                       errwidth=2, 
                       dodge=0.45, 
                       ax=ax)
    ax = sns.stripplot(x='Response_name', 
                       y='Value', 
                       hue='Response_type', 
                       hue_order=['EV','Predicted ICE','Actual ICE'],   
                       palette={'Actual ICE': (205/255, 0, 26/255), 
                                'Predicted ICE': (186/255, 51/255, 168/255), 
                                'EV':(34/255, 89/255, 149/255)},
                       alpha=0.3, 
                       data=error, 
                       dodge=0.45,
                       ax=ax)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:], labels[3:], bbox_to_anchor=(0.9,-0.4), ncol=3)
    ax = set_labels(ax, {'title': 'Predicted ICE Response', 'ylabel': get_units(chy[5:]), 'xlabel': ''})
    ax = adjust_font_sizes(ax, {'ticklabels': 16, 'title': 20, 'axlabels': 20, 'legend': 18})
#    ax.tick_params(axis='x', rotation=90)

#%% regression: no difference
ylist = ['Max_11HEAD0000THACRA',
         'Max_11CHST0000THACRC']
xlist = ['Weight']
subset = table.query('Speed==48 and Type==\'EV\'')
subset['Series_1'] = subset['Series_1'].replace(np.nan, 0)
#subset = table.loc[table.query('Series_2==1')['Pair']]
        
#xlist = ['Weight']
#subset = table.query('Series_2==1')
for chx in xlist:
    for chy in ylist:
        if chx==chy: continue
        x = arrange_by_group(subset, features[chx], 'Series_1')
        y = arrange_by_group(subset, features[chy], 'Series_1')
        if len(x)==0 or len(y)==0: continue
        x, y = match_keys(x, y)
        if len(x)==0 or len(y)==0: continue
        match_groups(x,y)
#        r2 = [rho(x[k], y[k]) for k in x]
#        if max(r2)<0.3 and min(r2)>-0.3: continue
        
        fig, ax = plt.subplots()
        ax = plot_scatter(ax, x, y)
        ax = set_labels(ax, {'xlabel': chx, 
                             'ylabel': chy, 
                             'legend': {'bbox_to_anchor': (1,1)},
                             'title': 'R=' + str(corr(pd.concat(list(x.values())), pd.concat(list(y.values()))))[:6]})
        ax = adjust_font_sizes(ax, {'ticklabels': 18, 'axlabels': 20, 'legend': 18, 'title': 24})
        plt.show()
        plt.close(fig)
