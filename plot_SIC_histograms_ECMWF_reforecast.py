#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:23:42 2019

@author: mcmcgraw
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy
import datetime
from datetime import datetime, timedelta, time
from scipy import signal
import scipy.stats as stats
import pandas as pd
import os
import csv

#Marie C. McGraw
#Atmospheric  Science, University of Washington
#Updated 7-23-2019

#Test basic VRILE examination in ECMWF. Load all netCDF files in directory (ECMWF reforecasts on SIPN grid) and concatenate in time
model_name = 'ecmwfsipn'
model_type = 'reforecast'
filepath = '/home/disk/sipn/nicway/data/model/{model_name}/{model_type}/sipn_nc_agg'.format(model_name=model_name,model_type=model_type)
filename = xr.open_mfdataset(filepath+'/*.nc',concat_dim='init_time')
#print(filename)
region_names = filename.region_names

#Notice that "Extent" has dimensions of [init_time x ensemble x fore_time x nregions]
#time:
#init_time: forecast initialization time. 1x/month, on the first of the month
#ensemble: ensemble number (51 total)
#fore_time: forecast time. Days since init_time (up to 215 days)
#nregions: regions (0 is panArctic, 1-14 are regions)
extent = filename.Extent
init_time = filename.init_time

#Get a list of dates for the forecast days
start_date = np.array(init_time[0].values).astype('datetime64[D]')
init_dates = pd.DatetimeIndex(np.array(init_time.values))
#
forecast_time = filename.fore_time #in nanoseconds (seriously? ugh)
multiplier = 86400000000000
forecast_days = forecast_time.values #covnert from nanoseconds
forecast_days = forecast_days/multiplier
forecast_dates = start_date
#print(forecast_dates)
for idate in np.arange(0,len(forecast_days)):
    idelta = start_date + np.timedelta64(forecast_days[idate]*multiplier,'D')
    forecast_dates = np.append(forecast_dates,idelta)

#print(forecast_dates)
#TIME = pd.DatetimeIndex(forecast_dates)

#We want to focus on JJAS
mon_sel = [6,7,8,9]  #looking at JJAS
mon_sel_str = 'JJAS'
day_sep = 2
no_days = 2*day_sep + 1
#region_sel = 0
##Create a climatology--remove average Jan 1 from all Jan 1, etc
TIME_all = np.array([])
#print(TIME_all)

        
for i in np.arange(0,len(init_time)):
    start_date_i = np.array(init_time[i].values).astype('datetime64[D]')
    forecast_dates_i = time(0,0) #represent forecast initialization with 0
    for jdate in np.arange(0,len(forecast_days)):
        idelta = start_date_i + np.timedelta64(forecast_days[jdate]*multiplier,'D')
        forecast_dates_i = np.append(forecast_dates_i,idelta)
        
    TIME_all = np.append(TIME_all,forecast_dates_i) #unraveled TIME
    
#now delete forecast initialization days
TIME_adj = np.delete(TIME_all,np.arange(0,288*216,216))
TIME = pd.DatetimeIndex(TIME_adj)
#now we can remove seasonal cycle from extent
SIE_seas_cyc = np.array([])
N = len(TIME)
E = extent.shape[1] #number of ensembles
#E = 5
#gap_days = 60 #forecast for up to 60 days
gap_day_sel = [7,14,21,28,35,42,49,56]
#Create Kara-Laptev, Barents-Kara, and East-Sib/Beaufort/Chukchi Seas
region_names = np.append(region_names,['Barents-Kara','Kara-Laptev','East-Siberian-Beaufort-Chukchi'])
extent_BK = extent[:,:,:,7] + extent[:,:,:,8]
extent_KL = extent[:,:,:,8] + extent[:,:,:,9]
extent_ESBC = extent[:,:,:,10] + extent[:,:,:,11] + extent[:,:,:,12]
extent_extras= np.stack((extent_BK,extent_KL,extent_ESBC),axis=3)
extent = np.concatenate((extent,extent_extras),axis=3)
#select_ind = np.arange(0,len(region_names))
#select_ind = [0,6,7,14,16,17]
select_ind = [0,6,16]
#select_ind = [0,6]
#
#region_names_sub = np.append(region_names_sub,(region_names[select_ind].values,'Kara-Laptev','East-Sib-Beaufort-Chukchi'))
#extent_sub = np.array([])
#extent_sub = np.append(extent_sub,(extent[:,:,:,select_ind],(extent[:,:,:,8]+extent[:,:,:,9]),(extent[:,:,:,10]+extent[:,:,:,11]+extent[:,:,:,12])))

#Create output directory
fpath_save = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/MODELS/{model_name}/{forecast}/'.format(model_name=model_name,forecast=model_type)
if not os.path.exists(fpath_save):
    os.makedirs(fpath_save)
for igap in gap_day_sel:
    print('lead time of {no_days} days'.format(no_days=igap))
    gap_days = igap
    #Loop through ensembles first
    for ireg in select_ind:
        region_sel = ireg
        rname = region_names[ireg]
        #rname = rname.replace(" ","-")
        print('region {region_name}'.format(region_name=rname))
        SIE_delta_ALL = np.array([])
        ##LOAD OBS
        #detrend_stat = 'NO_dt'
        region_name_fn = region_names[region_sel]
        #NSIDC_filename = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS/NSIDC_SIE_delta_{nodays}day_change_{region}_{seas}_{detrend_stat}.txt'.format(nodays=no_days,region=region_name_fn,seas=mon_sel_str,detrend_stat=detrend_stat)
        #NSIDC_SIE = np.genfromtxt(NSIDC_filename,unpack='True')
        #Create empty arrays for model stats
        MODEL_mean = np.array([])
        MODEL_var = np.array([])
        MODEL_skew = np.array([])
        MODEL_kurt = np.array([])
        
        pctile_vals = np.array([])
        
        
        for i_ens in np.arange(0,E,1):
            SIE_i_ens = np.squeeze(extent[:,i_ens,:,region_sel])
            print('ensemble no {no}'.format(no=i_ens+1))
            #
            SIE_i_ens_rs = SIE_i_ens.reshape((N,))
            #Remove mean from data
            SIE_i_ens_anom = SIE_i_ens_rs - np.nanmean(SIE_i_ens_rs)
            #Now we remove seasonal cycle
            SIE_rm_seas = np.array([])
            for iseas in np.arange(0,len(SIE_i_ens_anom)):
                idate = TIME[iseas]
                ind_sel = np.where((TIME.month==idate.month) & ((idate.day-day_sep<=TIME.day) & (TIME.day<=idate.day+day_sep)))
                SIE_rm_seas = np.append(SIE_rm_seas,(SIE_i_ens_anom[iseas,] - np.nanmean(SIE_i_ens_anom[ind_sel,])))
                
            SIE_anom_rs = SIE_rm_seas.reshape((len(init_time),len(forecast_time)))
            #Now, we calculate the 5-day changes in SIE
            SIE_delta_ens = np.array([])
            for i_init in np.arange(0,len(init_time)):
                SIE_i_init = SIE_anom_rs[i_init,:]
                day_init = init_dates[i_init]
                forecast_range = pd.date_range(day_init,periods=gap_days)
                #If any of the first 60 days fall in JJAS, continue
                if(any(np.isin(forecast_range.month,mon_sel))==True):
                    sel_ind = np.where(np.isin(forecast_range.month,mon_sel))
                    SIE_sel = SIE_i_init[sel_ind]
                    dates_sel = forecast_range[sel_ind] #selected dates
                    SIE_delta = np.array([])
                    for idelta in np.arange(day_sep,len(SIE_sel)-day_sep):
                        SIE_idelta = SIE_sel[idelta+day_sep] - SIE_sel[idelta-day_sep]
                        SIE_delta = np.append(SIE_delta,SIE_idelta)
                    
                    SIE_delta_ens = np.append(SIE_delta_ens,SIE_delta)
                    
            fig = plt.figure()
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
            ax.hist(SIE_delta_ens,bins=np.arange(-0.4,0.4,0.01),histtype=u'step',density=False,linewidth=3)
            ax.set_xlabel('{no_days} day change in SIC, 1993-2017 (10^6 km^2)'.format(no_days=2*day_sep+1))
            ax.set_ylabel('count')
            ax.set_title('{seas} SIC anomalies, ECMWF reforecast, {region_name}, {lead_day}day lead'.format(seas=mon_sel_str,
                         region_name=rname,lead_day=gap_days))            
            fname = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/'\
            'MODEL_{modelname}_{modeltype}_SIC_{no_days}day_change_{lead_day}dayLEAD_ensemble{ens_no}_'\
            '{seas}_{region}'.format(modelname=model_name,modeltype=model_type,
             no_days=no_days,lead_day=gap_days,ens_no=i_ens+1,seas=mon_sel_str,region=rname)
            plt.savefig(fname,format='png',dpi=600)
            
            if(i_ens==0):
                SIE_delta_ALL = SIE_delta_ens
            else:
                SIE_delta_ALL = np.vstack((SIE_delta_ALL,SIE_delta_ens))
            plt.close('all')
            
        fig2 = plt.figure()
        ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
        ax2.hist(np.transpose(SIE_delta_ALL),bins=np.arange(-0.4,0.4,0.01),histtype=u'step',density=False,linewidth=3)
        ax2.set_xlabel('{no_days} day change in SIC, 1993-2017 (10^6 km^2)'.format(no_days=2*day_sep+1))
        ax2.set_ylabel('count')
        ax.set_title('{seas} SIC anomalies, ECMWF reforecast, {region_name}, {lead_day} lead, all ensembles'.format(seas=mon_sel_str,
                     region_name=rname,lead_day=gap_days))
        fname2 = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/'\
            'MODEL_{modelname}_{modeltype}_SIC_{no_days}day_change_{lead_day}dayLEAD_ALL_ENS_'\
            '{seas}_{region}'.format(modelname=model_name,modeltype=model_type,
             no_days=no_days,lead_day=gap_days,seas=mon_sel_str,region=rname)
        plt.savefig(fname2,format='png',dpi=600)
        plt.close('all')
        #
        SIE_ens_mean = np.nanmean(np.transpose(SIE_delta_ALL))
        print(SIE_ens_mean.shape)
        
        fname_data_save = fpath_save+'{model_name}_{model_type}_SIE_delta_{nodays}day_change_{lead_day}dayLEAD_{region_name}_{seas}.txt'.format(model_name=model_name,
                                      lead_day=gap_days,model_type=model_type,nodays=no_days,region_name=region_name_fn,seas=mon_sel_str)
        np.savetxt(fname_data_save,np.array(np.transpose(SIE_delta_ALL)),fmt='%.9f')
        
        #Percentiles and stats
        pctile_values = [1,5,25,50,75,95,99]
        df_SIE_delta = pd.DataFrame(SIE_delta_ALL)
        fname_raw = fpath_save+'RAW_SIC_delta_{no_days}day_change_{lead_day}dayLEAD_{model_name}_ALL_ENS_{region}_{season}.csv'.format(no_days=no_days,
                                              lead_day=gap_days,model_name=model_name,region=region_name_fn,season=mon_sel_str)
        df_SIE_delta.to_csv(fname_raw)
        MODEL_pctiles = np.nanpercentile(SIE_delta_ALL,pctile_values,axis=1)
        MODEL_means = np.nanmean(SIE_delta_ALL,axis=1)
        MODEL_vars = np.nanvar(SIE_delta_ALL,axis=1)
        MODEL_skewx = stats.skew(SIE_delta_ALL,nan_policy='omit',axis=1)    
        MODEL_skew = np.ma.filled(MODEL_skewx,fill_value=99999999999)
        MODEL_kurt = stats.kurtosis(SIE_delta_ALL,nan_policy='omit',axis=1)
        #
        MODEL_stats = np.vstack((MODEL_means,MODEL_vars,MODEL_skew,MODEL_kurt))
        MODEL_stats_list = pd.DataFrame(np.transpose(MODEL_stats))
        print(MODEL_stats)
        #write files
        fname_stats = fpath_save+'STATS_SIC_{no_days}day_change_{lead_day}dayLEAD_{model_name}_ALL_ENS_{region}_{season}.csv'.format(no_days=no_days,
                                            lead_day=gap_days,model_name=model_name,region=region_name_fn,season=mon_sel_str)
        fname_pctiles = fpath_save+'PCTILES_SIC_{no_days}day_change_{lead_day}dayLEAD_{model_name}_ALL_ENS_{region}_{season}.csv'.format(no_days=no_days,
                                                lead_day=gap_days,model_name=model_name,region=region_name_fn,season=mon_sel_str)
        #
        MODEL_stats_list.to_csv(fname_stats,header=['MEAN','VAR','SKEW','KURTOSIS'])
        pd.DataFrame(np.transpose(MODEL_pctiles)).to_csv(fname_pctiles,header=['1','5','25','50','75','95','99'])
#    #with open(fname_stats,'w') as fstats:
#        filewriter = csv.writer(fstats,delimiter=',',quoting=csv.QUOTE_MINIMAL)
#        filewriter.writerow(['MEAN',MODEL_means.tolist()])
#        filewriter.writerow(['VARIANCE',MODEL_vars.tolist()])
#        filewriter.writerow(['SKEWNESS',MODEL_skew.tolist()])
#        filewriter.writerow(['KURTOSIS',MODEL_kurt.tolist()])
#        fstats.close()
     
#    with open(fname_pctiles,'w') as fpct:
#        fw2 = csv.writer(fpct,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
#        for ipct in np.arange(0,len(pctile_values)):
#            fw2.writerow([pctile_values[ipct],MODEL_pctiles[ipct,:].tolist()])
#        fpct.close()

