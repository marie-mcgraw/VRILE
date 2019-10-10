#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:23:48 2019

@author: mcmcgraw
"""

import xarray as xr
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, date
import seaborn as sns
import matplotlib.pyplot as plt

#Load model SIE data and group by region
filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
model_name = 'ecmwfsipn'
model_type = 'reforecast'
day_change = 5
max_lead = 30
month_ind = [6,7,8,9]
seas_str = 'JJAS'
lead_skip = 3
filename = 'MOVING_{model_name}_{model_type}_d_SIC_{day_change}day_change_lead_time_1{max_lead}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                                                                                                                                 model_type=model_type,
                                                                                                                                 day_change=day_change,
                                                                                                                                 max_lead=max_lead)

ds_SIC_all = pd.read_csv(filepath+filename)
regions = ds_SIC_all['region']
region_names = ds_SIC_all['region'].unique().tolist()
#Loop through regions 
for reg_sel_ind in np.arange(0,len(region_names)):
#reg_sel_ind = 7
    region_sel = region_names[reg_sel_ind]
    print('now running {region}'.format(region=region_sel))
    d_SIC_reg = ds_SIC_all.groupby(['region'])
    d_SIC_ireg = d_SIC_reg.get_group(region_sel)
    d_SIC_ireg['I (init date)'] = pd.to_datetime(d_SIC_ireg['I (init date)'])
    init_date = pd.DatetimeIndex(d_SIC_ireg['I (init date)'])
    d_SIC_ireg['V (valid date)'] = pd.to_datetime(d_SIC_ireg['V (valid date)'])
    valid_date = pd.DatetimeIndex(d_SIC_ireg['V (valid date)'])
    valid_dates_sel = valid_date.month.isin(month_ind)
    dSI = d_SIC_ireg['d_SIC (V - I)']
    dSI_sel = dSI[valid_dates_sel]
    #Now go as function of lead time
    lead_days = valid_date[valid_dates_sel] - init_date[valid_dates_sel]
    SIE_lead = pd.DataFrame(columns=['d_SIE','lead_days'])
    SIE_lead.loc[:,'d_SIE'] = dSI_sel
    SIE_lead.loc[:,'lead_days'] = lead_days
    #print(SIE_lead)
    #Group by lead day
    #poo = pd.Timedelta(days=5)
    #print(poo)
    SIE_lead_days = SIE_lead.groupby(['lead_days'])
    #print(lead_days)
    lead_days_vec = np.arange(5,max_lead+1,lead_skip)
    #print(lead_days_vec)
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,1.9,1.9])
    medianprops = {'color': 'blue', 'linewidth': 2}
    boxprops = {'color': 'black','linestyle':'-', 'linewidth': 3}
    whiskerprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    capprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    flierprops = {'color': 'black', 'marker': 'o', 'markersize': 1, 'markeredgewidth': 2}
    for iday in np.arange(0,len(lead_days_vec)):
        lead_days_groups = SIE_lead_days.get_group(pd.Timedelta(days=lead_days_vec[iday]))
        SIE_by_lead = lead_days_groups['d_SIE']
        ax1.boxplot(SIE_by_lead.dropna().values, positions=[iday],flierprops=flierprops,whis=[5,95],
                   medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops,
                   capprops=capprops)
        
    #ax1.set_ylim([-0.2,0.2])
    ax1.set_xlim([-0.5,len(lead_days_vec)])
    ax1.set_xticks(np.arange(0,len(lead_days_vec)))
    ax1.set_xticklabels(lead_days_vec)
    ax1.set_xlabel('Forecast lead time (days)',fontsize=12)
    ax1.set_ylabel('5-day change in sea ice extent (10^6 km^2)',fontsize=12)
    ax1.set_title('5-day change in sea ice extent, {region}, {model_name} {model_type}, {month_str}, all ensembles'.format(region=region_sel,
                                                                                              model_name=model_name,
                                                                                              model_type=model_type,
                                                                                              month_str=seas_str),fontsize=13)
    #plt.show()
    fname_1 = "/home/disk/sipn/mcmcgraw/figures/VRILE/model_d_SIE_vs_lead_time/"\
    "{model_name}_{model_type}_{region}_d_SIE_vs_lead_time_BOXPLOT_{max_lead}day_max_lead_"\
    "{seas_str}_all_ens_skip_{lead_skip}days.png".format(model_name=model_name,
     model_type=model_type,region=region_sel,max_lead=max_lead,seas_str=seas_str,lead_skip=lead_skip)
    fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
    plt.close('all')
