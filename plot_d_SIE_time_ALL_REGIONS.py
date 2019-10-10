#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:37:41 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime, timedelta
import statsmodels.api as sm

def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvar]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()
    
    
model_name = 'ecmwfsipn'
model_type = 'reforecast'
no_ens = 25 #25 ensemble members
no_day_change = 5 #looking at 5 day changes
day_change = no_day_change
max_lead = 30

delta_lead_days = np.arange(no_day_change,max_lead+1,1)
delta_first_days = np.arange(1,max_lead+2-no_day_change,1)
no_forecast_periods = len(delta_lead_days)

filepath_load = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
filename_full = filepath_load+'MOVING_{model_name}_{model_type}_d_SIC_{d_days}day_change_lead_time_{lead_days}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                               model_type=model_type,d_days=no_day_change,
                               lead_days=no_day_change*no_forecast_periods)

file_full = pd.read_csv(filename_full)
regions_list = file_full['region'].unique().tolist()
valid_dates_list = file_full['V (valid date)'].unique().tolist()

sie_regions = file_full.groupby(['region'])
#ireg = 1
for ireg in np.arange(0,len(regions_list)):
    region_name = regions_list[ireg]
    sie_ireg = sie_regions.get_group(region_name)
    sie_valid = sie_ireg.groupby(['V (valid date)'])
    sie_val_mean = sie_ireg.groupby(['V (valid date)']).mean()
    sie_val_med = sie_ireg.groupby(['V (valid date)']).median()
    sie_val_min = sie_ireg.groupby(['V (valid date)']).min()
    sie_val_max = sie_ireg.groupby(['V (valid date)']).max()
    sie_val_5th = sie_ireg.groupby(['V (valid date)']).quantile(0.05)
    sie_val_95th = sie_ireg.groupby(['V (valid date)']).quantile(0.95)
    
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.9,0.9])
    ax1.grid(True)
    minmax = ax1.fill_between(np.arange(0,len(sie_val_mean)),sie_val_min['d_SIC (V - I)'].values,sie_val_max['d_SIC (V - I)'].values,color='xkcd:sky blue')
    ptiles = ax1.fill_between(np.arange(0,len(sie_val_mean)),sie_val_5th['d_SIC (V - I)'].values,sie_val_95th['d_SIC (V - I)'].values,color='xkcd:salmon')
    meanvals = ax1.plot(sie_val_mean['d_SIC (V - I)'].values,'k')
    ax1.set_xlim([0,len(valid_dates_list)])
    ax1.set_xticks(np.arange(0,len(valid_dates_list),312))
    ax1.set_xticklabels(valid_dates_list[0::312],rotation=30,ha='right',position=(0,0),fontsize=9.5)
    ax1.set_xlabel('Valid date',fontsize=12)
    ax1.set_ylabel('5-day change in SIE (10^6 km^2)',fontsize=12)
    ax1.set_title(region_name)
    ax1.legend((minmax,ptiles),('Min / Max','5th / 95th pctiles'),bbox_to_anchor=(1.335,0.91))
    #plt.show()
    fname_1 = '/home/disk/sipn/mcmcgraw/figures/VRILE/d_SIE_time_series/model/{model_name}_{model_type}_{region}_{max_lead}day_lead_{day_change}day_change_raw.png'.format(model_name=model_name,
                                                                              model_type=model_type,region=region_name,max_lead=max_lead,
                                                                              day_change=day_change)
    fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
    
    #Now, smooth with a rolling filter
    rolling_interval = 25*1 #25--one month
    mean_smooth = sie_val_mean.rolling(rolling_interval).mean()
    #mean_smooth
    #ds_val_mean
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1,0.1,0.9,0.9])
    ax2.grid(True)
    s_minmax = ax2.fill_between(np.arange(0,len(sie_val_mean)),sie_val_min['d_SIC (V - I)'].rolling(rolling_interval).mean(),sie_val_max['d_SIC (V - I)'].rolling(rolling_interval).mean(),color='xkcd:sky blue')
    s_ptiles = ax2.fill_between(np.arange(0,len(sie_val_mean)),sie_val_5th['d_SIC (V - I)'].rolling(rolling_interval).mean(),sie_val_95th['d_SIC (V - I)'].rolling(rolling_interval).mean(),color='xkcd:salmon')
    ax2.plot(mean_smooth['d_SIC (V - I)'].values,'k')
    ax2.set_xlim([0,len(valid_dates_list)])
    ax2.set_xticks(np.arange(0,len(valid_dates_list),312))
    ax2.set_xticklabels(valid_dates_list[0::312],rotation=30,ha='right',position=(0,0),fontsize=9.5)
    ax2.set_xlabel('Valid date',fontsize=12)
    ax2.set_ylabel('5-day change in SIE (10^6 km^2)',fontsize=12)
    ax2.set_title(region_name+' smoothed')
    ax2.legend((s_minmax,s_ptiles),('Min / Max','5th / 95th pctiles'),bbox_to_anchor=(1.335,0.91))
    #plt.show()
    fname_2 = '/home/disk/sipn/mcmcgraw/figures/VRILE/d_SIE_time_series/model/{model_name}_{model_type}_{region}_{max_lead}day_lead_{day_change}day_change_SMOOTHED_{rolling_interval}_days.png'.format(model_name=model_name,
                                                                              model_type=model_type,region=region_name,max_lead=max_lead,
                                                                              day_change=day_change,rolling_interval=rolling_interval)
    fig2.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')
    #pause(5)
    plt.close('all')