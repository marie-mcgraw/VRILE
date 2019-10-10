#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:01:51 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from datetime import datetime, date, timedelta
import scipy.stats as stats
from itertools import compress

filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
filepath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/'
model_name = 'ecmwfsipn'
model_type = 'reforecast'
day_change = 5
max_lead = 30

filename = '{model_name}_{model_type}_d_SIC_{day_change}day_change_lead_time_{max_lead}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                                                                                                                                 model_type=model_type,
                                                                                                                                 day_change=day_change,
                                                                                                                                 max_lead=max_lead)
#filename2 = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/ecmwfsipn_reforecast_d_SIC_5day_change_lead_time_30days_ALL_REGIONS_ALL_ENS.csv'
ds_SIC_all = pd.read_csv(filepath+filename)

regions = ds_SIC_all['region']
region_names = ['panArctic','East Greenland Sea','Barents Sea','Central Arctic',
                'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
region_names_obs = ['panArctic','EastGreenlandSea','BarentsSea','CentralArctic',
                'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
filepath_obs = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS/'  #observed sea ice extent data
##open_mfdataset(foo+'/*.nc') opens all nc files in specific directory
init_date_ALL = pd.to_datetime(ds_SIC_all['I (init date)'])
valid_date_ALL = pd.to_datetime(ds_SIC_all['V (valid date)'])
lead_time = valid_date_ALL - init_date_ALL
ds_SIC_all['Lead time'] = lead_time
lead_time_vec = np.arange(5,31,5)
mon_sel = [6,7,8,9]
##Information for obs
#fname_time_obs = 'NSIDC_SIE_delta_TIME_5day_change_ALL_NO_dt.csv'
#time_obs = pd.read_csv(filepath_obs+fname_time_obs)
#obs_year = time_obs['year']
#obs_month = time_obs['month']
#obs_mon_sel = obs_month.isin(mon_sel)
#OBS
mon_names = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
#fname_obs = 'NSIDC_SIE_delta_{day_change}day_change_{region}_ALL_NO_dt.txt'.format(day_change=day_change,
#                         region=region_names_obs[ireg])
#SIC_obs = pd.read_csv(filepath_obs+fname_obs)
#d_SIC_obs_mon = pd.DataFrame(columns=mon_names)
#d_SIC_obs_sel = compress(SIC_obs,obs_mon_sel)   
valid_mon = valid_date_ALL.dt.month
seas_str = 'JJAS'
mon_sel_ind = np.isin(valid_mon,mon_sel)
d_SIC = ds_SIC_all['d_SIC (V - I)']

d_SIC_mon_sel = d_SIC.iloc[mon_sel_ind]
ds_SIC_all_mon_sel = ds_SIC_all.iloc[mon_sel_ind]
#boxplot 
medianprops = {'color': 'blue', 'linewidth': 2}
boxprops = {'color': 'black','linestyle':'-', 'linewidth': 3}
whiskerprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
capprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
flierprops = {'color': 'black', 'marker': 'o', 'markersize': 1, 'markeredgewidth': 2}

pct5_df = pd.DataFrame(columns=region_names)  
plot_color_names = ['xkcd:sky blue',
                    'xkcd:salmon',
                    'xkcd:olive green',
                    'xkcd:dark red',
                    'xkcd:ochre',
                    'xkcd:light purple']
    #ireg = 0
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
    
for ireg in np.arange(0,len(region_names)):
    boxplot_pos_vec = np.arange(0,6)
    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    pct5_ireg = np.array([])
    for ilead in lead_time_vec:
        d_SIC_test = ds_SIC_all_mon_sel['d_SIC (V - I)'].where((ds_SIC_all_mon_sel['region'] == region_names[ireg]) & (ds_SIC_all_mon_sel['Lead time']==timedelta(days=ilead.astype(float))))
        #
        d_SIC_test_plot = d_SIC_test[~np.isnan(d_SIC_test)]
        med = np.median(d_SIC_test_plot)
        pct5 = np.percentile(d_SIC_test_plot,5)
        pct5_ireg = np.append(pct5_ireg,pct5)
        box_pos = ilead/5
        ax1.boxplot(d_SIC_test_plot,positions=[box_pos.astype(float)],
                                               medianprops=medianprops,
                                               boxprops=boxprops,
                                               whiskerprops=whiskerprops,
                                               capprops=capprops,
                                               flierprops=flierprops)
        ax1.set_xlim((0.5,6.5))
        if ireg == 0:
            ax1.set_ylim((-0.7,0.7))
            ax1.text(box_pos.astype(float),0.6,round(med,2))
        else:
            ax1.set_ylim((-0.35,0.35))
            ax1.text(box_pos.astype(float),0.3,round(med,2))
       
    
    ax1.set_xticks(np.arange(1,7))    
    ax1.set_xticklabels(lead_time_vec)
    ax1.set_xlabel('Forecast lead time (days)',fontsize=12)    
    ax1.set_ylabel('Change in SIE (10^6 km^2)',fontsize=12)
    title_str = '{region}, {season} ({model_name}, {model_type})'.format(region=region_names[ireg],
                 season=seas_str,model_name=model_name,model_type=model_type)
    ax1.set_title(title_str,fontsize=12)
    fname_1 = filepath_save+'Lead_time_vs_SIE_{region}_{season}_{model_name}_{model_type}.png'.format(region=region_names[ireg],
                                              season=seas_str,model_name=model_name,
                                              model_type=model_type)
    
    plt.savefig(fname_1,format='png',dpi=600)
    reg_name = region_names[ireg]
    pct5_df[reg_name] = pct5_ireg


    
    ax2.plot(lead_time_vec,pct5_ireg,'o-',markersize=3,color=plot_color_names[ireg])
    
ax2.legend(region_names,bbox_to_anchor=(1.025,1.025),fontsize=9.5)
ax2.set_xlabel('Forecast lead time (days)',fontsize=12)
ax2.set_ylabel('Change in SIE (10^6 km^2)',fontsize=12 )
ax2.set_title('Mean Magnitude of 5th percentile in SIE',fontsize=12 )
fname_2 = filepath+'fifth_pctile_vs_lead_time_ALL_REGIONS_{season}_{model_name}_{model_type}.png'.format(season=seas_str,
                                                          model_name=model_name,model_type=model_type)
plt.savefig(fname_2,format='png',dpi=600)