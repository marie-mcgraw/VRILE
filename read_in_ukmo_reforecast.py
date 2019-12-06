#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:44:12 2019

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


# Import forecast data (2017-19)

# In[38]:


model_name = 'ukmo'
model_type = 'reforecast'
filepath = '/home/disk/sipn/nicway/data/model/{model_name}/{model_type}/sipn_nc_agg/'.format(model_name=model_name,
                                              model_type=model_type)
filenames = xr.open_mfdataset(filepath+'/*.nc',concat_dim='init_time')
print(filenames)


# Create Kara-Laptev and E-Sib/Beauf/Chukchi

# In[39]:


region_names = filenames.region_names
region_names = np.append(region_names,['Kara-Laptev','East-Siberian-Beaufort-Chukchi'])
init_times = filenames.init_time
forecast_times = filenames.fore_time
extent = filenames.Extent
##chunk sizes in dimensions of [init_time x ensemble x fore_time x region]
chunk_sizes = filenames['Extent'].shape
extent_KL = extent[:,:,:,8] + extent[:,:,:,9]
extent_ESBC = extent[:,:,:,10] + extent[:,:,:,11] + extent[:,:,:,12]
extent_extras= np.stack((extent_KL,extent_ESBC),axis=3)
extent = np.concatenate((extent,extent_extras),axis=3)


# For now, we'll define VRILEs as the 5th percentile events of 5-day changes

# In[40]:


no_ens = len(filenames.ensemble) ##no. of ensemble members
no_day_change = 5 ##looking at 5 day changes
no_forecast_periods = len(forecast_times)


# initialize our output. Since SIE is a time series, we'll use Pandas and a DataFrame. For now, we will track initialization date, valid date, the actual SIE, lead time (in days, this will be a timedelta object), 5-day change in SIE (this will be recorded for the center day), ensemble number, and region.

# In[42]:


d_SIC_ALL_ens = pd.DataFrame(columns=["I (init date)",
                                      "V (valid date)",
                                      "V_mon (valid date month)",
                                      "V_yr (valid date year)",
                                      "SIE",
                                      "lead time (V - I)",
                                      "d_SIE (V - I)",
                                      "ensemble",
                                      "region"])


# Loop through each region, then each forecast time and just calculcate d_SIE and add to dataframe

# In[43]:


##Create integers for each region
##I could probably write this better with more uses of groupby. 
reg_sel = np.arange(0,17)
#reg_sel = [0,6,7,14,15,16]
##Outer loop will go through each region
for ireg in reg_sel:
    #ireg = 0
    region_name = region_names[ireg]
    region_select = ireg
    print(region_name)
    ##Next loop will go through each init time
    for itime in np.arange(0,len(init_times)):
    #itime = 0
        init_times_df = pd.DatetimeIndex(init_times.values)
        init_select = init_times_df[itime]#.to_dataset()
        check_yr = pd.to_datetime(init_select).year
        #if check_yr != 2018:
            #print('not the right yr')
            #continue
        #print(init_select)
        print('reading time ',init_times[itime])
        ##We'll create another DataFrame inside this loop; we'll append it 
        ##to the big DataFrame outside of this loop.
        d_SIC_lead_time = pd.DataFrame({"I (init date)":pd.Series(init_select).repeat(len(forecast_times)*no_ens),
                                    "V (valid date)":"",
                                    "V_mon (valid date month)":"",
                                    "V_yr (valid date year)":"",
                                    "SIE":"",
                                    "lead time (days)":"",
                                    "d_SIE (V - I)":"",
                                    "ensemble":"",
                                    "region":""})
        ##Now, we loop through our ensemble members
        for iens in np.arange(0,no_ens):
            #iens = 0
            ##Keep track of the correct indices so we don't have to append ad infitum
            save_ind = iens*(no_forecast_periods-4) + np.arange(0,no_forecast_periods-4)
            #print('ensemble no ',iens)
            #        d_SIC_lead_time['ensemble'].iloc[ens_ind] = np.tile(iens,no_forecast_periods*len(init_times))
            #Subset our sea ice extent by init_tim, ensemble no., and region
            I_test = extent[iens,itime,:,region_select]
            ##since we're doing 5-day means, our first and last 2 dates aren't included
            ind_select = np.arange(2,no_forecast_periods-2) 
            min_range = ind_select - 2
            max_range = ind_select + 2
            ##Here's where we actually calculate that 5-day change in SIE
            delta_extent = I_test[max_range] - I_test[min_range]
            d_SIC_lead_time['d_SIE (V - I)'].iloc[save_ind] = delta_extent
            ##Now, we get the dates that correspond to our valid date and number of lead days
            forecast_dates = ind_select.astype('timedelta64[D]')
            date_change = pd.Series(init_select).repeat(len(forecast_dates)) + forecast_dates
            d_SIC_lead_time['V (valid date)'].iloc[save_ind] = pd.to_datetime(date_change.values)
            d_SIC_lead_time['V_mon (valid date month)'].iloc[save_ind] = pd.to_datetime(date_change.values).month
            d_SIC_lead_time['V_yr (valid date year)'].iloc[save_ind] = pd.to_datetime(date_change.values).year
            ##We want to save lead time as a time delta, not a date
            d_SIC_lead_time["lead time (days)"].iloc[save_ind] = pd.to_timedelta(forecast_dates).days
            ##This is just for saving files, because Python is 0-indexed but our ensemble no isn't
            ens_no = iens + 1
            ##Save info about our ensemble, region, and raw SIE data
            d_SIC_lead_time['ensemble'].iloc[save_ind] = np.tile(ens_no,len(delta_extent))
            d_SIC_lead_time['region'].iloc[save_ind] = np.tile(region_name,len(delta_extent))
            d_SIC_lead_time['SIE'].iloc[save_ind] = I_test[ind_select]
            #d_SIC_lead_time
        if itime == 0:
            df_ALL_init = d_SIC_lead_time
        else:        
            df_ALL_init = df_ALL_init.append(d_SIC_lead_time)
        #    
    if ireg == 0:
        d_SIC_ALL_ens = df_ALL_init
        #filename_full = filepath_save+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
        #               model_type=model_type,d_days=no_day_change)
        #d_SIC_ALL_ens.to_csv(filename_full)
    else:
        d_SIC_ALL_ens = d_SIC_ALL_ens.append(df_ALL_init)
    #filename_full = filepath_save+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
    #               model_type=model_type,d_days=no_day_change)
    #d_SIC_ALL_ens.to_csv(filename_full)


# In[36]:


filepath_save = '/home/disk/sipn/mcmcgraw/data/VRILE/'
filename_full = filepath_save+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                       model_type=model_type,d_days=no_day_change)
d_SIC_ALL_ens.to_csv(filename_full)
