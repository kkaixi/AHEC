# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:34:28 2018
Get the variance for the time series
@author: tangk
"""
from PMG.COM.timeseries import estimate_ts_variance
import pandas as pd
from PMG.read_data import initialize
import numpy as np

directory = 'P:\\Data Analysis\\Projects\\AHEC EV\\'
cutoff = range(100,1600)
n = 1
channels=['10CVEHCG0000ACXD',
          '11HEAD0000THACXA',
          '11HEAD0000THACYA',
          '11SPIN0100THACXC',
          '11CHST0000THACXC',
          '11CHST0000THACYC',
          '11SPIN1200THACXC',
          '11PELV0000THACXA',
          '11PELV0000THACYA',
          '11FEMRLE00THFOZB',
          '11FEMRRI00THFOZB',
          '11SEBE0000B3FO0D',
          '11SEBE0000B6FO0D',
          '11ILACLE00THFOXA',
          '11ILACRI00THFOXA',
          '11ACTBLE00THFOXB',
          '11ACTBRI00THFOXB',
          '11NIJCIPTETH00YX',
          '11NIJCIPTFTH00YX',
          '11NIJCIPCFTH00YX',
          '11NIJCIPCETH00YX',
          '11CHSTLEUPTHDSXB',
          '11CHSTRIUPTHDSXB',
          '11CHSTLELOTHDSXB',
          '11CHSTRILOTHDSXB',
          '11NECKUP00THFOZA',
          '11ACTBLE00THFORB',
          '11ACTBRI00THFORB',
          '11CLAVLEOUTHFOXA',
          '11CLAVLEINTHFOXA']
#channels = ['11HEAD0000THACXA','11CHST0000THACXC','11PELV0000THACXA']
#table, t, chdata = initialize(directory,channels,cutoff,query='ID11==\'TH\' and Subset==\'CONTROL\'')

def get_avg(data):
    #data is a pd.series
    return np.nanmean(np.vstack(data.values),axis=0)

# average the chest responses
chest = ['11CHSTLEUPTHDSXB',
         '11CHSTRIUPTHDSXB',
         '11CHSTLELOTHDSXB',
         '11CHSTRILOTHDSXB']
chdata['11CHST0000THDSXB'] = chdata[chest].apply(get_avg,axis=1).apply(lambda x: tuple(x)).apply(lambda x: np.array(x))
channels.append('11CHST0000THDSXB')

pairs = {'Cruze': ['TC17-025','TC15-035'],
          'Camry': ['TC15-162','TC17-028'],
          'Civic':['TC15-155','TC17-203'],
          'Sentra':['TC16-205','TC17-206'],
          'Explorer':['TC11-233','TC11-234']}
lp, up  = estimate_ts_variance(chdata,tclist=pairs,channels=channels,n=n,method='diff')
bounds = pd.concat([lp.mean().rename('negative'),up.mean().rename('positive')],axis=1,levels=['a','b'])
bounds.to_csv(directory + 'ts_variance.csv')


#for ch in channels:
#    for p in pairs:
#        plt.plot(t,chdata[ch][pairs[p][0]],label=pairs[p][0])
#        plt.plot(t,chdata[ch][pairs[p][1]],label=pairs[p][1])
#        plt.legend()
#        plt.title(p + ' ' + ch)
#        plt.show()