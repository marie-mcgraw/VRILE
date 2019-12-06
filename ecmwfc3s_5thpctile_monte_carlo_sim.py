#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:35:50 2019

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
import random

##Clear workspace before running 
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

#Load model sea ice extent (SIE) data    
filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/'
model_name = 'ecmwfsipn'
model_type = 'reforecast'
day_change = 5
max_lead = 30

filename = filepath+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                               model_type=model_type,d_days=day_change)
#filename2 = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/ecmwfsipn_reforecast_d_SIC_5day_change_lead_time_30days_ALL_REGIONS_ALL_ENS.csv'
ds_SIC_all = pd.read_csv(filename)
ds_SIC_all = ds_SIC_all.dropna()
pctile_select = 5 #look at 5th percentile
regions = ds_SIC_all['region']
#region_names = ['panArctic','East Greenland Sea','Barents Sea','Central Arctic',
 #               'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
region_names = ds_SIC_all['region'].unique().tolist()

#Create column names for output, which will be stored in Pandas dataframes
column_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Region']
mon_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#initialize dataframe for 5th percentile from model data
model_SIE_p5_MC = pd.DataFrame(columns=column_names)
model_SIE_p95_MC = pd.DataFrame(columns=column_names)
model_SIE_median_MC = pd.DataFrame(columns=column_names)
obs_SIE_p5 = pd.DataFrame(columns=column_names)
obs_SIE_p95 = pd.DataFrame(columns=column_names)
obs_SIE_median = pd.DataFrame(columns=column_names)
obs_minus_model_SIE_p5_MC = pd.DataFrame(columns=column_names)
obs_minus_model_SIE_p95_MC = pd.DataFrame(columns=column_names)
obs_minus_model_SIE_median_MC = pd.DataFrame(columns=column_names)
obs_p5_count_in_model_MC = pd.DataFrame(columns=column_names)
obs_p95_count_in_model_MC = pd.DataFrame(columns=column_names)
obs_median_count_in_model_MC = pd.DataFrame(columns=column_names)
#track 25th and 75th percentiles as well
model_SIE_p25_MC = pd.DataFrame(columns=column_names)
model_SIE_p75_MC = pd.DataFrame(columns=column_names)
obs_minus_model_SIE_p25_MC = pd.DataFrame(columns=column_names)
obs_minus_model_SIE_p75_MC = pd.DataFrame(columns=column_names)
#track min vRILE? another day
#Now, load data from NSIDC (observations)
#region_names_obs = ['panArctic','EastGreenlandSea','BarentsSea','CentralArctic',
#                'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
#region_names_obs = [iname.replace(' ','') for iname in region_names]
obs_name = 'NSIDC_0079'
obs_type = 'sipn_nc_yearly_agg'
obs_filename = filepath+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=obs_name,
                       model_type=obs_type,d_days=day_change)
SIE_obs = pd.read_csv(obs_filename)
SIE_obs = SIE_obs.dropna()
#
n_samples = 10**2 #number of monte carlo experiments
pctile_select = 5 #looking at 5th/95th pctile
pctile_select2 = 25 #looking at 25th/75th pctile
for ireg in np.arange(0,len(region_names)):
#    #ireg = 6
    print('now running {region}'.format(region=ireg))
    iname = region_names[ireg]
#    #Select only d_SI values for that region
    region_sel = regions.index.where(regions==iname)
    ds_SIC_ireg = ds_SIC_all.iloc[~np.isnan(region_sel),:]
    #Convert initialization and valid dates to datetime objects
    ds_SIC_ireg['I (init date)'] = pd.to_datetime(ds_SIC_ireg['I (init date)'])
    init_date2 = ds_SIC_ireg['I (init date)']
    ds_SIC_ireg['V (valid date)'] = pd.to_datetime(ds_SIC_ireg['V (valid date)'])
    valid_date = ds_SIC_ireg['V (valid date)']
    #dSI = ds_SIC_ireg['d_SIC (V - I)']
    #Select observed data for specified region
    SIC_obs = pd.read_csv(filepath_obs+fname_obs)
    ##Filter obs--do things change?
    filt = True
    filt_app = 'NO_FILT' #used for filenames
    if filt == True:
        SIC_obs = SIC_obs.rolling(5).mean()
        filt_app = 'FILT'
    #Now, separate by month
    lead_sel = 7
    ds_SIC_lead_group = ds_SIC_ireg.groupby(['lead time (days)'])
    ds_SIC_lead_sel = ds_SIC_lead.get_group(lead_sel)
    dSI = ds_SIC_lead_sel['d_SIE (V - I)']
    for imon in np.arange(1,13):
        mon_sel_ind = valid_date.index.where(valid_date.dt.month == imon) #select correct month for model
        d_SIC_mon_sel = dSI.where(~np.isnan(mon_sel_ind)) #model data corresponding to that month--all ensembles
        d_SIC_mon_sel = d_SIC_mon_sel.dropna()#drop NaNs so we actually know sample size
        imon_name = mon_names[imon-1] #name of month
        print('now calculating {month}'.format(month=imon_name))
        #Obs--select obs for each month
        obs_sel_ind = obs_month.index.where(obs_month == imon)
        obs_sel_ind = obs_sel_ind[0:-5]
        d_SIC_obs_sel = SIC_obs[~np.isnan(obs_sel_ind)]
        obs_p5 = np.nanpercentile(d_SIC_obs_sel,pctile_select)
        obs_p95 = np.nanpercentile(d_SIC_obs_sel,100-pctile_select)
        obs_p50 = np.nanmedian(d_SIC_obs_sel)
        obs_SIE_p5.loc[ireg,imon_name] = obs_p5
        obs_SIE_p95.loc[ireg,imon_name] = obs_p95
        obs_SIE_median.loc[ireg,imon_name] = obs_p50
        obs_SIE_p5.loc[ireg,'Region'] = iname
        obs_SIE_p95.loc[ireg,'Region'] = iname
        obs_SIE_median.loc[ireg,'Region'] = iname
        #Randomly sample model to match obs
        p5_MC = np.array([])
        p50_MC = np.array([])
        p95_MC = np.array([])
        p25_MC = np.array([])
        p75_MC = np.array([])
        N_obs = len(d_SIC_obs_sel)
        N_mod = len(d_SIC_mon_sel)
        for isamp in np.arange(0,n_samples):
            rand_sel = random.sample(range(0,N_mod),N_obs) #randomly select N_obs values from model
            rand_sel.sort()
            d_rand_mod = d_SIC_mon_sel.iloc[rand_sel]
            model_p5 = np.nanpercentile(d_rand_mod,pctile_select)
            model_p95 = np.nanpercentile(d_rand_mod,100-pctile_select)
            model_p25 = np.nanpercentile(d_rand_mod,pctile_select2)
            model_p75 = np.nanpercentile(d_rand_mod,100-pctile_select2)
            model_p50 = np.nanmedian(d_rand_mod)
            p5_MC = np.append(p5_MC,model_p5)
            p95_MC = np.append(p95_MC,model_p95)
            p25_MC = np.append(p25_MC,model_p25)
            p75_MC = np.append(p75_MC,model_p75)
            p50_MC = np.append(p50_MC,model_p50)
        p5_count = p5_MC[np.where(p5_MC < obs_p5)] #how often is model VRILE stronger than obs? 
        p95_count = p95_MC[np.where(p95_MC > obs_p95)] #how often is model VRIGE stronger than obs? 
        p50_count = p50_MC[np.where(abs(p50_MC) > abs(obs_p50))] #how often is model median 5day SIE change larger than obs
        p5_mean = np.nanmean(p5_MC)
        #p5_min = np.nanmin(p5_MC)
        #p5_max = np.nanmax(p5_MC)
        p95_mean = np.nanmean(p95_MC)
        #p95_min = np.nanmin(p95_MC)
        #p95_max = np.nanmax(p95_MC)
        p25_mean = np.nanmean(p25_MC)
        p75_mean = np.nanmean(p75_MC)
        model_SIE_p5_MC.loc[ireg,imon_name] = p5_mean
        model_SIE_p95_MC.loc[ireg,imon_name] = p95_mean
        model_SIE_median_MC.loc[ireg,imon_name] = np.nanmean(p50_MC)
        obs_minus_model_SIE_p5_MC.loc[ireg,imon_name] = obs_p5 - p5_mean
        obs_minus_model_SIE_p95_MC.loc[ireg,imon_name] = obs_p95 - p95_mean
        obs_minus_model_SIE_median_MC.loc[ireg,imon_name] = obs_p50 - np.nanmean(p50_MC)
        obs_p5_count_in_model_MC.loc[ireg,imon_name] = 100*(len(p5_count)/len(p5_MC))
        obs_p95_count_in_model_MC.loc[ireg,imon_name] = 100*(len(p95_count)/len(p95_MC))
        obs_median_count_in_model_MC.loc[ireg,imon_name] = 100*(len(p50_count)/len(p50_MC))
        model_SIE_p25_MC.loc[ireg,imon_name] = p25_mean
        model_SIE_p75_MC.loc[ireg,imon_name] = p75_mean
        #regions
        model_SIE_p5_MC.loc[ireg,'Region'] = iname
        model_SIE_p95_MC.loc[ireg,'Region'] = iname
        model_SIE_median_MC.loc[ireg,'Region'] = iname
        model_SIE_p25_MC.loc[ireg,'Region'] = iname
        model_SIE_p75_MC.loc[ireg,'Region'] = iname
        obs_minus_model_SIE_p5_MC.loc[ireg,'Region'] = iname
        obs_minus_model_SIE_p95_MC.loc[ireg,'Region'] = iname
        obs_minus_model_SIE_median_MC.loc[ireg,'Region'] = iname
        obs_p5_count_in_model_MC.loc[ireg,'Region'] = iname
        obs_p95_count_in_model_MC.loc[ireg,'Region'] = iname    
        obs_median_count_in_model_MC.loc[ireg,'Region'] = iname
#        
#model_SIE_p5_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "{model_name}_{model_type}_5th_pctile_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
#                       #
#model_SIE_p95_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "{model_name}_{model_type}_95th_pctile_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
#model_SIE_median_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "{model_name}_{model_type}_median_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
#obs_SIE_p5.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "NSIDC_5th_pctile_RESAMPLED_MC.csv")
#obs_SIE_p95.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "NSIDC_95th_pctile_RESAMPLED_MC.csv")
#obs_SIE_median.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "NSIDC_median_RESAMPLED_MC.csv")
#obs_minus_model_SIE_p5_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "obs_minus_{model_name}_{model_type}_5th_pctile_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
#obs_minus_model_SIE_p95_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "obs_minus_{model_name}_{model_type}_95th_pctile_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
#obs_minus_model_SIE_median_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "obs_minus_{model_name}_{model_type}_median_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
##
#obs_p5_count_in_model_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "COUNT_{model_name}_{model_type}_stronger_than_obs_5th_pctile_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
#obs_p95_count_in_model_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "COUNT_{model_name}_{model_type}_stronger_than_obs_95th_pctile_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
#obs_median_count_in_model_MC.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
#                       "COUNT_{model_name}_{model_type}_stronger_than_obs_median_RESAMPLED_MC.csv".format(model_name=model_name,
#                        model_type=model_type))
