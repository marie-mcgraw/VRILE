#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:22:43 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime, timedelta

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
    
#load all model files
model_name = 'ecmwfsipn'
model_type = 'reforecast'
filepath = '/home/disk/sipn/nicway/data/model/{model_name}/{model_type}/sipn_nc_agg/'.format(model_name=model_name,
                                              model_type=model_type)
filenames = xr.open_mfdataset(filepath+'/*.nc',concat_dim='init_time')
print(filenames)

region_names = filenames.region_names
region_names = np.append(region_names,['Kara-Laptev','East-Siberian-Beaufort-Chukchi'])
init_times = filenames.init_time
forecast_times = filenames.fore_time
extent = filenames.Extent
extent_KL = extent[:,:,:,8] + extent[:,:,:,9]
extent_ESBC = extent[:,:,:,10] + extent[:,:,:,11] + extent[:,:,:,12]
extent_extras= np.stack((extent_KL,extent_ESBC),axis=3)
extent = np.concatenate((extent,extent_extras),axis=3)

no_ens = 25 #25 ensemble members
no_day_change = 5 #looking at 5 day changes
max_lead = 30

delta_lead_days = np.arange(no_day_change,max_lead+1,1)
delta_first_days = np.arange(1,max_lead+2-no_day_change,1)
no_forecast_periods = len(delta_lead_days)

d_SIC_ALL_ens = pd.DataFrame(columns=["I (init date)","V (valid date)","d_SIC (V - I)","ensemble","region"])
#reg_sel = [0,6,7,14,15,16]
reg_sel = np.arange(0,17)
#for ireg in np.arange(0,len(region_names)):
for ireg in reg_sel:
    region_name = region_names[ireg]
    region_select = ireg
    print(region_name)
    #for iens in np.arange(0,no_ens):
    #df_ALL_ens = pd.DataFrame(columns=["I (init date)","V (valid date)","d_SIC (V - I)","ensemble","region"])
    #df_ALL_ens = pd.DataFrame(columns=["I (init date)","V (valid date)","d_SIC (V - I)"])
    #for iens in np.arange(0,2):
    #Create empty dataframe for all init times
    #df_ALL_full = pd.DataFrame(columns=["I (init date)","V (valid date)","d_SIC (V - I)"])
    #test_XR = xr.DataArray(
    #        np.empty((3000, 25, 15)),
    #                 coords = ([('time',np.arange(0,3000)), 
    #                            ('ensemble',np.arange(0,25)), 
    #                            ('region',np.arange(0,15))]))
    #for iens in np.arange(0,no_ens):
    #for itime in np.arange(0,len(init_times)):
    for itime in np.arange(0,len(init_times)):
    #itime = 0
        init_select = init_times[itime]
        print(init_select)
        d_SIC_lead_time = pd.DataFrame({"I (init date)":np.tile(init_select,no_forecast_periods*no_ens),
                                    "V (valid date)":"","d_SIC (V - I)":"","ensemble":"","region":""})
        for iens in np.arange(0,no_ens):
            save_ind = iens*no_forecast_periods + np.arange(0,no_forecast_periods)
            print('ensemble no ',iens)
    #        d_SIC_lead_time['ensemble'].iloc[ens_ind] = np.tile(iens,no_forecast_periods*len(init_times))
            I_test = extent[itime,iens,:,region_select]
            delta_extent = I_test[delta_lead_days] - I_test[delta_first_days]
            d_SIC_lead_time['d_SIC (V - I)'].iloc[save_ind] = delta_extent
            ens_no = iens + 1
            #d_SIC_lead_time['ensemble'].iloc[save_ind] = np.tile(ens_no,len(delta_extent))
            for i in np.arange(0,len(delta_lead_days)):
                ival = init_select + pd.Timedelta(delta_lead_days[i],unit='D')
                write_ind = save_ind[i]
                d_SIC_lead_time['V (valid date)'].iloc[write_ind] = pd.to_datetime(ival.values).date()
                d_SIC_lead_time['ensemble'].iloc[write_ind] = iens + 1
                d_SIC_lead_time['region'].iloc[write_ind] = region_name
                
        if itime == 0:
            df_ALL_init = d_SIC_lead_time
        else:        
            df_ALL_init = df_ALL_init.append(d_SIC_lead_time)
            
    if ireg == 0:
        d_SIC_ALL_ens = df_ALL_init
    else:
        d_SIC_ALL_ens = d_SIC_ALL_ens.append(df_ALL_init)
        

filepath_save = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
filename_full = filepath_save+'{model_name}_{model_type}_d_SIC_{d_days}day_change_lead_time_{lead_days}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                               model_type=model_type,d_days=no_day_change,
                               lead_days=no_day_change*no_forecast_periods)
d_SIC_ALL_ens.to_csv(filename_full)
#    if iens == 0:
#        df_ALL_ens = df_ALL_init
#    else:
#        df_ALL_ens = df_ALL_ens.append(df_ALL_init)
#    init_select = init_times[itime]
#    #create a dataframe for each lead time; we'll keep adding to this
#    d_SIC_lead = pd.DataFrame({"I (init date)":np.tile(init_select,no_forecast_periods*no_ens),
#                                "V (valid date)":"","d_SIC (V - I)":""})
#    #now loop through each ensemble member
    
