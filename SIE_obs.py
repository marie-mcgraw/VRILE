#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:35:28 2019

@author: mcmcgraw
"""

"""
Created on Thu Aug  8 17:22:43 2019
Updated on Wed Oct 23 2019
@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime, timedelta

##This code clears all the variables from the workspace; can help avoid memory errors
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
    
##load all NSIDC files. use NSIDC extent that has been regridden on SIPN grid and divided into regions
model_name = 'NSIDC_0081'
model_type = 'sipn_nc_yearly_agg'
filepath = '/home/disk/sipn/nicway/data/obs/{model_name}/{model_type}/'.format(model_name=model_name,
                                              model_type=model_type)
filenames = xr.open_mfdataset(filepath+'/*.nc',concat_dim='time')
print(filenames)

##We'll create our aggregate regions, Kara-Laptev and East Siberian-Beaufort-Chukchi
region_names = filenames.region_names
region_names = np.append(region_names,['Kara-Laptev','East-Siberian-Beaufort-Chukchi'])
#init_times = filenames.init_time
time_obs = filenames.time
extent = filenames.Extent
###chunk sizes in dimensions of [init_time x ensemble x fore_time x region]
chunk_sizes = filenames['Extent'].shape
extent_KL = extent[:,8] + extent[:,9]
extent_ESBC = extent[:,10] + extent[:,11] + extent[:,12]
extent_extras= np.stack((extent_KL,extent_ESBC),axis=1)
extent = np.concatenate((extent,extent_extras),axis=1)
#
###For now, we define VRILEs as extreme 5-day changes in sea ice extent (SIE)
no_day_change = 5 ##looking at 5 day changes
no_forecast_periods = chunk_sizes[0]
#
###initialize our output. Since SIE is a time series, we'll use Pandas and a DataFrame.
###For now, we will track initialization date, valid date, the actual SIE,
###lead time (in days, this will be a timedelta object), 5-day change in SIE 
###(this will be recorded for the center day), ensemble number, and region.
d_SIC_ALL_obs = pd.DataFrame(columns=["V (valid date)",
                                      "V_mon (valid date month)",
                                      "SIE",
                                      "d_SIC (V - I)",
                                      "ensemble",
                                      "region"])
###Create integers for each region
###I could probably write this better with more uses of groupby. 
reg_sel = np.arange(0,17)
###Outer loop will go through each region
for ireg in reg_sel:
#ireg = 0
    region_name = region_names[ireg]
    region_select = ireg
    print(region_name)
    ###Next loop will go through each init time
    ###We'll create another DataFrame inside this loop; we'll append it 
    ###to the big DataFrame outside of this loop.
    d_SIC_lead_time = pd.DataFrame({"V (valid date)":"",
                                "V_mon (valid date month)":"",
                                "SIE":"",
                                "d_SIC (V - I)":"",
                                "ensemble":np.tile("obs",no_forecast_periods),
                                "region":""})
    ###Now, we loop through our ensemble members
    ###Keep track of the correct indices so we don't have to append ad infitum
    save_ind = np.arange(0,no_forecast_periods-4)
    #print('ensemble no ',iens)
    ##        d_SIC_lead_time['ensemble'].iloc[ens_ind] = np.tile(iens,no_forecast_periods*len(init_times))
    ##Subset our sea ice extent by init_tim, ensemble no., and region
    I_test = extent[:,region_select]
    ###since we're doing 5-day means, our first and last 2 dates aren't included
    ind_select = np.arange(2,no_forecast_periods-2) 
    min_range = ind_select - 2
    max_range = ind_select + 2
    ###Here's where we actually calculate that 5-day change in SIE
    delta_extent = I_test[max_range] - I_test[min_range]
    d_SIC_lead_time['d_SIC (V - I)'].iloc[save_ind] = delta_extent
    ###Now, we get the dates that correspond to our valid date and number of lead days
    time_save = time_obs.isel(time=slice(ind_select[0],ind_select[-1]+1))
    d_SIC_lead_time['V (valid date)'].iloc[save_ind] = pd.to_datetime(time_save.values).date
    d_SIC_lead_time['V_mon (valid date month)'].iloc[save_ind] = pd.to_datetime(time_save.values).month
    ###Save info about our region, and raw SIE data
    d_SIC_lead_time['region'].iloc[save_ind] = np.tile(region_name,len(delta_extent))
    d_SIC_lead_time['SIE'].iloc[save_ind] = I_test[ind_select]
    #        
    ##if itime == 0:
    ##    df_ALL_init = d_SIC_lead_time
    ##else:        
    ##    df_ALL_init = df_ALL_init.append(d_SIC_lead_time)
    ##    
    if ireg == 0:
        d_SIC_ALL_obs = d_SIC_lead_time
    else:
        d_SIC_ALL_obs = d_SIC_ALL_obs.append(d_SIC_lead_time)
#
#
###Save our final dataframe as a .csv file
filepath_save = '/home/disk/sipn/mcmcgraw/data/VRILE/'
filename_full = filepath_save+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                       model_type=model_type,d_days=no_day_change)
d_SIC_ALL_obs.to_csv(filename_full)
print('saved!!')