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
         'Min_11CHST0000THACXC': 'Chest Acx',
         'Min_11CHST0000H3ACXC': 'Chest Acx',
         'Min_13CHST0000HFACXC': 'Chest Acx',
         'Min_11CHST0000xxACXC': 'Chest Acx',
         'Max_11CHST0000xxACRC': 'Chest AcR',
         'Min_11SPIN0100THACXC': 'Upper Spine Acx',
         '11HEAD0000THACXA': 'Head Acx',
         '11FEMRLE00THFOZB': 'Left Femur Fz',
         'Max_11NECKUP00THFOZA': 'Upper Neck Fz',
         'Max_11NECKUP00H3FOZA': 'Upper Neck Fz',
         'Max_13NECKUP00HFFOZA': 'Upper Neck Fz',
         'Max_11NECKUP00xxFOZA': 'Upper Neck Fz',
         'TH': 'THOR',
         'H3': 'Hybrid III',
         'HF': 'Hybrid III Female',
         'Min_11PELV0000xxACXA': 'Pelvis Acx'}
              
rename = partial(rename, names=names)
#%% time series overlays w/o highlight
plot_channels = ['10CVEHCG0000ACXD',
                 '10SIMELE00INACXD',
                 '11HEAD0000THACXA',
                 '11NECKUP00THFOZA',
                 '11SPIN0100THACXC',
                 '11CHST0000THACXC',
                 '11PELV0000THACXA',
                 '11FEMRLE00THFOZB',
                 '11FEMRRI00THFOZB']
grouped = table.table.query_list('Model',['SOUL','SOUL EV','OPTIMA','OPTIMA PHEV']).groupby('Model')
for grp in grouped:
    subset = grp[1]
    subset['TC'] = subset.index
    for ch in plot_channels:
        x = arrange_by_group(subset, chdata[ch], 'TC')
#        x['Electric (MY 2016)'] = x.pop('EV')
#        x['Conventional (MY 2015)'] = x.pop('ICE')
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, t, x)
        ax = set_labels(ax, {'title': ' '.join((grp[0], rename(ch))), 'xlabel': 'Time [s]', 'ylabel': get_units(ch), 'legend': {'bbox_to_anchor': [1,1]}})
        ax = adjust_font_sizes(ax, {'title': 24, 'axlabels': 20, 'legend': 20, 'ticklabels': 18})
        plt.show()
        plt.close(fig)
#%%
      
plot_channels = [['Max_11HICR0015THACRA','Max_11HICR0015H3ACRA'],
                 ['Max_11HEAD003STHACRA','Max_11HEAD003SH3ACRA'],
                 ['Min_11HEAD0000THACXA','Min_11HEAD0000H3ACXA'],
                 ['Max_11NECKUP00THFOZA','Max_11NECKUP00H3FOZA'],
                 ['Max_11CHST003STHACRC','Max_11CHST003SH3ACRC'],
                 ['Min_11CHST0000THACXC','Min_11CHST0000H3ACXC']]
subset = table.query('Pair_name==\'FUSION\'')
for ch in plot_channels:
    feat_subset = features.loc[subset.index, ch]
    if len(ch)>1:
        feat_subset = feat_subset.dropna(how='all').apply(np.nansum, axis=1).rename('ch').abs()
    else:
        feat_subset = feat_subset.squeeze().rename('ch').abs()
    x = pd.concat((subset, feat_subset), axis=1)
    fig, ax = plt.subplots()
    ax = sns.barplot(x='ID11', y='ch', hue='Type', data=x)
    ax = set_labels(ax, {'title': rename(ch[0]), 'ylabel': get_units(ch[0]), 'xlabel': ''})
    ax = adjust_font_sizes(ax, {'title': 24, 'axlabels': 20, 'ticklabels': 20})
    
#%% mpl bar plots of individual pairs
plot_channels = ['Max_11HICR0015THACRA',
                 'Max_11HICR0015H3ACRA',
                 'Max_11HEAD003STHACRA',
                 'Max_11HEAD003SH3ACRA',
                 'Min_11HEAD0000THACXA',
                 'Min_11HEAD0000H3ACXA',
                 'Max_11NECKUP00THFOZA',
                 'Max_11NECKUP00H3FOZA',
                 'Max_11CHST003STHACRC',
                 'Max_11CHST003SH3ACRC',
                 'Min_11CHST0000THACXC',
                 'Min_11CHST0000H3ACXC',
                 'Min_11SPIN0100THACXC',
                 'Max_11SEBE0000B6FO0D',
                 'Max_13HEAD003SHFACRA',
                 'Min_13HEAD0000HFACXA',
                 'Max_13NECKUP00HFFOZA',
                 'Max_13CHST003SHFACRC',
                 'Min_13CHST0000HFACXC',
                 'Max_13SEBE0000B6FO0D']
plot_channels = ['Max_11HEAD003STHACRA',
                 'Max_11NECKUP00THFOZA',
                 'Max_11CHST003STHACRC',
                 'Max_11CHST003SH3ACRC']

for d in dummies:
    for s in ['Series_1']:
        subset = table.query(s + '==1 and ID11==\'{}\''.format(d))
        pairs = subset['Pair_name'].unique()
        codes = letters[:len(pairs)]
        for ch in plot_channels:
            ch_x = features.loc[subset.index, ch].abs()
            if ch_x.isna().all(): continue
            x = pd.concat([subset, ch_x], axis=1)
            fig, ax = plt.subplots()
            ax = sns.barplot(x='Pair_name', y=ch, hue='Type_2', data=x)
#            ax.set_xticklabels(list(codes))
            ax = set_labels(ax, {'title': rename(ch), 'ylabel': get_units(ch), 'xlabel': 'Model'})
            ax = adjust_font_sizes(ax, {'title': 24, 'ticklabels': 20, 'axlabels': 20})
            plt.xticks(rotation=90)
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
s2['Series'] = '(Series 2)'
pairs = pairs.append(s2)
pairs['ID11'] = pairs['ID11'].apply(rename)
pairs['Label'] = pairs[['ID11','Series']].apply(lambda x: '\n'.join(x), axis=1)

r = re.compile('_\d\d')

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
    ax = sns.pointplot(x='Label', y='ch', data=pairs, join=False, ci='sd', err_style='bars', capsize=0.2, color='.25', errwidth=2, ax=ax)
    ax = sns.stripplot(x='Label', y='ch', data=pairs, color='.25', alpha=0.4, size=8, ax=ax, marker='*')
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
#    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], color=(0.5, 0.5, 0.5), alpha=0.2)
    ax = set_labels(ax, {'title': rename(title), 'xlabel': '', 'ylabel': 'Difference in ' + get_units(title)})
    ax = adjust_font_sizes(ax, {'ticklabels': 20, 'title': 24, 'ylabel': 20})
    ax.set_ylim([i*2 for i in ax.get_ylim()])
    ax.set_xlim([-1, 2.5])
    plt.locator_params(axis='y', nbins=10)
#    plt.savefig(directory + title + '.png', dpi=600, bbox_inches='tight')
    plt.show()

#%% sns catplot 2
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

    ax = sns.boxplot(x='Label', y='ch', data=pairs, linewidth=1, 
                     boxprops={'alpha': 0.5}, width=0.4, sym='')
    ax = sns.stripplot(x='Label', y='ch', data=pairs, color='.25', ax=ax)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
#    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1], color=(0.5, 0.5, 0.5), alpha=0.2)
    ax = set_labels(ax, {'title': rename(title), 'xlabel': '', 'ylabel': get_units(title)})
    ax = adjust_font_sizes(ax, {'ticklabels': 20, 'title': 24, 'ylabel': 20})
#    plt.savefig(directory + title + '.png', dpi=600, bbox_inches='tight')
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


subset = table.query('Type==\'ICE\' and Speed==48')
for chx in xlist:
    for chy in ylist:
        if chx==chy: continue
        rsq = r2(features.loc[subset.index, chx], features.loc[subset.index, chy])
        x = features.loc[subset.index, [chx, chy]]
        x['Series'] = ['Series 1' if subset.at[i,'Series_1']==1 else 'Series 2' for i in x.index]
        fig, ax = plt.subplots()
        ax = sns.scatterplot(x=chx, y=chy, hue='Series', data=x, ax=ax)
        ax = set_labels(ax, {'xlabel': chx, 'ylabel': chy, 'title': 'R2={}'.format(str(rsq)[:5])})
        ax = adjust_font_sizes(ax, {'axlabels': 20, 'ticklabels': 18, 'title': 24})
        plt.show()
        plt.close(fig)

#%% create a linear regression model and use it to predict responses based on weight/partner weight to 
# assess how much the two account for observed differences in the response
from sklearn.linear_model import LinearRegression
ylist = ['Max_11HEAD0000THACRA',
#         'Max_11NECKUP00THFOZA',
         'Max_11CHST0000THACRC',
         'Min_11CHST0000THACXC',
         'Min_11PELV0000THACXA',
         'Max_13HEAD0000HFACRA',
         'Min_13HEAD0000HFACXA',
         'Min_13CHST0000HFACXC']
xlist = [['Weight']]



subset = table.query('Type==\'ICE\' and Speed==48')
subset_test = table.query('Type==\'EV\' and Speed==48')
for chx in xlist:
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
        error.append(pd.DataFrame({'Value': np.concatenate((y_pred, y_test, np.squeeze(y))),
                                   'Response_type': np.concatenate((np.tile('Predicted',len(y_pred)), 
                                                              np.tile('Actual',len(y_test)),
                                                              np.tile('ICE', len(y)))),
                                   'Response_name': chy}))
    error = pd.concat(error)
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.pointplot(x='Response_name', y='Value', hue='Response_type', hue_order=['ICE','Predicted','Actual'], data=error, join=False, ci='sd', err_style='bars', capsize=0.2, errwidth=2, dodge=0.5, ax=ax)
    #ax = sns.boxplot(x='Response', y='Error', boxprops={'alpha': 0.5}, data=error)
    ax = sns.stripplot(x='Response_name', y='Value', hue='Response_type', hue_order=['ICE','Predicted','Actual'], alpha=0.3, data=error, dodge=0.5)
    ax.axhline(0,linewidth=1,color='k')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(chx)

#%% same as above but backwards
from sklearn.linear_model import LinearRegression
ylist = ['Max_11HEAD0000THACRA',
#         'Max_11NECKUP00THFOZA',
         'Max_11CHST0000THACRC',
         'Min_11CHST0000THACXC',
         'Min_11PELV0000THACXA',
         'Max_13HEAD0000HFACRA',
         'Min_13HEAD0000HFACXA',
         'Min_13CHST0000HFACXC']
xlist = [['Weight']]



subset = table.query('Type==\'ICE\' and Speed==48')
subset_test = table.query('Type==\'EV\' and Speed==48')
for chx in xlist:
    error = []
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
#        err = y_pred-y_test
        error.append(pd.DataFrame({'Value': np.concatenate((y_pred, y, np.squeeze(y_test))),
                                   'Response_type': np.concatenate((np.tile('Predicted',len(y_pred)), 
                                                              np.tile('Actual',len(y)),
                                                              np.tile('EV', len(y_test)))),
                                   'Response_name': chy}))
    error = pd.concat(error)
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.pointplot(x='Response_name', y='Value', hue='Response_type', hue_order=['EV','Predicted','Actual'], data=error, join=False, ci='sd', err_style='bars', capsize=0.2, errwidth=2, dodge=0.5, ax=ax)
    #ax = sns.boxplot(x='Response', y='Error', boxprops={'alpha': 0.5}, data=error)
    ax = sns.stripplot(x='Response_name', y='Value', hue='Response_type', hue_order=['EV','Predicted','Actual'], alpha=0.3, data=error, dodge=0.5)
    ax.axhline(0,linewidth=1,color='k')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(chx)
#%%
from sklearn.linear_model import LassoLars, Lars
from sklearn.preprocessing import StandardScaler
ylist = ['Max_11CHST0000THACRC',
         'Min_11CHST0000THACXC']
drop = ['Max_11CHST0000THACRC',
        'Max_11CHST003STHACRC',
        'Min_11CHST0000THACXC',
        'Auc_11CHST003STHACRC',
        'Min_11SPIN1200THACXC',
        'Min_11CHSTRIUPTHDSXB']
subset = table.query('Type==\'ICE\' and Speed==48')
ss = StandardScaler()
for chy in ylist:
    y = features.loc[subset.index, chy]
    y = y[~y.isna()]
    x = features.drop(chy, axis=1).loc[y.index].dropna(how='all',axis=1)
    x = x.drop([i for i in drop if i in x.columns], axis=1)
    x = pd.DataFrame(ss.fit_transform(x), index=x.index, columns=x.columns)
    drop_cols = x.isna().sum()
    drop_cols = drop_cols[drop_cols>len(y)//2]
    x = x.drop(drop_cols.index, axis=1)
    cols_with_na = x.isna().any()
    cols_with_na = cols_with_na[cols_with_na]
    # fill in missing numbers
    for col in cols_with_na.index:
        x[col] = x[col].replace(np.nan, x[col].mean())
#    model = LassoLars(max_iter=1000)
    model = Lars(n_nonzero_coefs=1)
    model = model.fit(x, y)
    coefs = pd.Series(model.coef_, index=x.columns)
    coefs = coefs[coefs!=0]
    print(chy)
    print(coefs)
    print(model.score(x,y))
    print('\n')
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
