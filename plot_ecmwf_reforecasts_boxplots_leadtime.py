#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:49:56 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from datetime import datetime, date
import scipy.stats as stats

filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
model_name = 'ecmwfsipn'
model_type = 'reforecast'
day_change = 5
max_lead = 30

filename = 'MOVING_{model_name}_{model_type}_d_SIC_{day_change}day_change_lead_time_1{max_lead}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                                                                                                                                 model_type=model_type,
                                                                                                                                 day_change=day_change,
                                                                                                                                 max_lead=max_lead)
#filename2 = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/ecmwfsipn_reforecast_d_SIC_5day_change_lead_time_30days_ALL_REGIONS_ALL_ENS.csv'
ds_SIC_all = pd.read_csv(filepath+filename)

regions = ds_SIC_all['region']
#region_names = ['panArctic','East Greenland Sea','Barents Sea','Central Arctic',
 #               'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
region_names = ds_SIC_all['region'].unique().tolist()
#region_names_obs = ['panArctic','SeaofOkhotsk'EastGreenlandSea','BarentsSea','CentralArctic',
#                'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
region_names_obs = [iname.replace(' ','') for iname in region_names]
filepath_obs = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS/'  #observed sea ice extent data
##open_mfdataset(foo+'/*.nc') opens all nc files in specific directory
#init_date_ALL = pd.to_datetime(ds_SIC_all['I (init date)'])
#valid_date_ALL = pd.to_datetime(ds_SIC_all['V (valid date)'])
fname_time_obs = 'NSIDC_SIE_delta_TIME_5day_change_ALL_NO_dt.csv'
time_obs = pd.read_csv(filepath_obs+fname_time_obs)
obs_year = time_obs['year']
obs_month = time_obs['month']

for ireg in np.arange(0,len(region_names)):
    #ireg = 4
    region_sel = regions.index.where(regions==region_names[ireg])
    ds_SIC_ireg = ds_SIC_all.iloc[~np.isnan(region_sel),:]
    #Convert initialization and valid dates to datetime objects
    ds_SIC_ireg['I (init date)'] = pd.to_datetime(ds_SIC_ireg['I (init date)'])
    init_date2 = ds_SIC_ireg['I (init date)']
    ds_SIC_ireg['V (valid date)'] = pd.to_datetime(ds_SIC_ireg['V (valid date)'])
    valid_date = ds_SIC_ireg['V (valid date)']
    dSI = ds_SIC_ireg['d_SIC (V - I)']
    #lead_time = valid_date - init_date2
    mon_ind = np.arange(1,13)
    mon_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    d_SIC_mon = pd.DataFrame(columns=mon_names)
    
    ###OBS
    fname_obs = 'NSIDC_SIE_delta_{day_change}day_change_{region}_ALL_NO_dt.txt'.format(day_change=day_change,
                             region=region_names_obs[ireg])
    SIC_obs = pd.read_csv(filepath_obs+fname_obs)
    d_SIC_obs_mon = pd.DataFrame(columns=mon_names)
    region_name = region_names[ireg]
    #sort by month of Valid date
    for imon in mon_ind:
     #   print(imon)
        mon_sel_ind = valid_date.index.where(valid_date.dt.month == imon)# & valid_date.dt.year > 1999)
        #select only d_SIC values where valid date is in that month
        d_SIC_mon_sel = dSI.where(~np.isnan(mon_sel_ind))
        #print(d_SIC_mon_sel)
        save_ind = imon-1
        mon_str = mon_names[imon-1]
        #save_ind = str(save_ind)
        #d_SIC_mon[mon_str] = d_SIC_mon_sel - np.nanmean(d_SIC_mon_sel)
        d_SIC_mon[mon_str] = d_SIC_mon_sel
        #OBS
        obs_sel_ind = obs_month.index.where(obs_month == imon)
        obs_sel_ind = obs_sel_ind[0:-5]
        d_SIC_obs_sel = SIC_obs[~np.isnan(obs_sel_ind)]
        #d_SIC_obs_sel = d_SIC_obs_sel - np.nanmean(d_SIC_obs_sel)
        d_SIC_obs_sel = pd.DataFrame(d_SIC_obs_sel)
        d_SIC_obs_mon = pd.concat([d_SIC_obs_mon,d_SIC_obs_sel],ignore_index=True,axis=1)
    #        if imon == 1:
    #            d_SIC_obs_mon = d_SIC_obs_sel
    #        else:
    #            d_SIC_obs_mon = np.hstack((d_SIC_obs_mon,d_SIC_obs_sel))
        
    medianprops = {'color': 'blue', 'linewidth': 2}
    boxprops = {'color': 'black','linestyle':'-', 'linewidth': 3}
    whiskerprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    capprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    flierprops = {'color': 'black', 'marker': 'o', 'markersize': 1, 'markeredgewidth': 2}
    fig1 = plt.figure()
    ax1 = sns.boxplot(data=d_SIC_mon,flierprops=flierprops)
    ax1.set_ylabel('change in SIE (10^6 km^2)')
    titname_1 = region_names[ireg]+', Model'
    ax1.set_title(titname_1)
    fname_save_1 = '/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_{region}_{dday}day_change_{lday}day_lead.png'.format(model_name=model_name,
                                                           model_type=model_type,region=region_names[ireg],
                                                           dday=day_change,lday=max_lead)
    plt.savefig(fname_save_1,format='png',dpi=600)
    #input("Press Enter to continue...")
    
    #OBS
    fig2 = plt.figure()
    ax2 = sns.boxplot(data=d_SIC_obs_mon.iloc[:,12:24])
    ax2.set_ylabel('change in SIE (10^6 km^2)')
    ax2.set(xticklabels=mon_names)
    #ax1.set_xlabel_ticks(mon_names)
    titname_2 = region_names[ireg]+', NSIDC'
    ax2.set_title(titname_2)
    fname_save_2 = '/home/disk/sipn/mcmcgraw/figures/VRILE/NSIDC_{region}_{dday}day_change_{lday}day_lead.png'.format(region=region_names[ireg],
                                                           dday=day_change,lday=max_lead)
    plt.savefig(fname_save_2,format='png',dpi=600)
    
    
    #Make histograms for each month
    month_str = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG',
                 'SEP','OCT','NOV','DEC']
    xmon_ind = np.arange(0,12)
    d_SIC_obs_mon_plot = d_SIC_obs_mon.iloc[:,12:24]
    #p_vals = pd.DataFrame(columns=region_names)
    p_mon = np.array([])
    #D_scores = pd.DataFrame(columns=region_names)
    D_mon = np.array([])
    for xmon in xmon_ind:
          d_SIC_hist = d_SIC_obs_mon_plot.iloc[:,xmon]
          d_SIC_hist_model = d_SIC_mon.iloc[:,xmon]
          
          [D,p] = stats.ks_2samp(d_SIC_hist,d_SIC_hist_model)
          #p_vals[ireg,imon] = p
          p_mon = np.append(p_mon,p)
          #D_scores[ireg,imon] = D
          D_mon = np.append(D_mon,D)
    #      print('D = ',D)
    #      print('p val is ',p)
    #      if abs(p) >= 0.05:
    #          print('not significantly different')
          bin_max_mod = np.amax(d_SIC_hist_model)+0.1
          bin_max_obs = np.amax(d_SIC_hist)+0.1
          bin_max = max(bin_max_mod,bin_max_obs)
          bin_min_mod = np.amin(d_SIC_hist_model)-0.1
          bin_min_obs = np.amin(d_SIC_hist)-0.1
          bin_min = min(bin_min_mod,bin_min_obs)
          bins = np.linspace(bin_min,bin_max,50)
          fig3 = plt.figure()
          ax3 = sns.kdeplot(d_SIC_hist_model,shade=False,color="r",label='model')
          ax3b = sns.kdeplot(d_SIC_hist,shade=False,color="b",label='obs')
          #fig3.legend(['model','obs'])
          plt.ylabel('count')
          plt.xlabel('5-day change in SIE')
          plt.xlim((-0.5,0.5))
          tit_str = '5 day change in SIE, {mon_str}, {region}'.format(mon_str=month_str[xmon],
                                          region=region_name)
          plt.title(tit_str)
          printstr = 'p value: {pv}'.format(pv=np.format_float_scientific(p,precision=2))
          if abs(p) >= 0.05:
              plt.text(-0.35,2,printstr)
          fname_save_3 = '/home/disk/sipn/mcmcgraw/figures/VRILE/monthly_distrib_model_vs_obs/distrib_obs_model_{mon}_{region}_{dday}day_change_{lday}day_lead.png'.format(mon=month_str[xmon],
                                                                           region=region_name,dday=day_change,lday=max_lead)
          plt.savefig(fname_save_3,format='png',dpi=600)
          
    if (ireg == 0):
        p_vals = p_mon
        D_scores = D_mon
    else:
        p_vals = np.vstack((p_vals,p_mon))
        D_scores = np.vstack((D_scores,D_mon))
    
    #    p_vals[region_name] = p_mon
    #    D_scores[region_name] = D_mon
    plt.close("all")
    
p_vals = np.transpose(p_vals)
D_scores = np.transpose(D_scores)

p_vals_PD = pd.DataFrame(data=p_vals,columns=region_names)
D_scores_PD = pd.DataFrame(data=D_scores,columns=region_names)