#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:55:20 2019

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
import cartopy.crs as ccrs
import cartopy.feature
#from mpl_toolkits.basemap import Basemap
#rcParams['savefig.bbox'] = 'tight'
#rcParams['savefig.pad_inches'] = 0.125

#Marie C. McGraw
#Atmospheric  Science, University of Washington
#Updated 7-23-2019

#def plot_Z500_anoms(lons,lats,z500):
    
#Test basic VRILE examination in ECMWF. Load all netCDF files in directory (ECMWF reforecasts on SIPN grid) and concatenate in time
model_name = 'ecmwfc3s'
#model_type = 'reforecast'
variable_name = 'Z500'
filepath = '/home/disk/sipn/mcmcgraw/data/atmospheric_data/{model_name}/{variable_name}/'.format(model_name=model_name,
                                                           variable_name=variable_name)
#Dimensions are [time x number x latitude x longitude]
#number is number of ensembles
#time is forecast valid date
filename = xr.open_mfdataset(filepath+'/*.nc',concat_dim='time')
ftime = filename.time
z500 = filename.z
lat = filename.latitude
lon = filename.longitude
s1,s2,s3,s4 = z500.shape

#Get a list of dates for the forecast days
valid_dates = pd.DatetimeIndex(np.array(ftime.values))

#Load ECMWF SIC data
model_name_SIE = 'ecmwfsipn'
model_type_SIE = 'reforecast'
no_ens = 25 #25 ensemble members
#Use binomial distribution to determine significance for sign agreement plots
for isign in np.arange(25,0,-1):
        pbin = stats.binom_test(isign,n=no_ens,p=0.5)
        if pbin >= 0.025:
            count_sig = isign+1
            break

no_day_change = 5 #looking at 5 day changes
day_change = no_day_change
max_lead = 30
mon_sel_ind = [6,7,8,9]
mon_str = 'JJAS'
delta_lead_days = np.arange(no_day_change,max_lead+1,1)
delta_first_days = np.arange(1,max_lead+2-no_day_change,1)
no_forecast_periods = len(delta_lead_days)

SIE_filepath_load = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
SIE_filename_full = SIE_filepath_load+'MOVING_{model_name}_{model_type}_d_SIC_{d_days}day_change_lead_time_{lead_days}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name_SIE,
                               model_type=model_type_SIE,d_days=no_day_change,
                               lead_days=no_day_change*no_forecast_periods)

SIE_file_full = pd.read_csv(SIE_filename_full)
SIE_regions_list = SIE_file_full['region'].unique().tolist()
SIE_valid_dates_list = SIE_file_full['V (valid date)'].unique().tolist()

SIE_region_groups = SIE_file_full.groupby(['region'])

#Identify lowest 5th pctile of SIE dates.  First sort by region and ensemble
#for ireg in SIE_regions_list:
ireg = 'Barents Sea'
print('now plotting {region}'.format(region=ireg))
region_sel_test = ireg
SIE_region_sel_test = SIE_region_groups.get_group(region_sel_test)
no_ens = 6
for iens in np.arange(0,no_ens):
    #iens = 0
    print('now plotting ensemble number {ens}'.format(ens=iens+1))
    ens_sel = iens+1
    SIE_ens_sel_test = SIE_region_sel_test.groupby(['ensemble']).get_group(ens_sel)
    #SIE_ens_sel_test = SIE_region_sel_test
    #Calculate 5th percentile
    pctile5 = SIE_ens_sel_test['d_SIC (V - I)'].quantile(0.05)
    valid_date_ens = SIE_ens_sel_test['V (valid date)']
    vdate_check = pd.DatetimeIndex(np.array(valid_date_ens.values))
    find_VRILE_days = SIE_ens_sel_test['d_SIC (V - I)'].where(SIE_ens_sel_test['d_SIC (V - I)'] <= pctile5)
    VRILE_dates = SIE_ens_sel_test['V (valid date)'].where(SIE_ens_sel_test['d_SIC (V - I)'] <= pctile5)
    #Now, grab Z500 for these dates
    VRILE_dates_Z500 = (valid_dates.isin(VRILE_dates) & valid_dates.month.isin(mon_sel_ind))
    valid_dates_summer = valid_dates.month.isin(mon_sel_ind)
    valid_dates_summer = valid_dates_summer.flatten()
    valid_dates_ind = np.asarray(np.where(VRILE_dates_Z500 == True))
    valid_dates_ind = valid_dates_ind.flatten()
    
    #Z500_select = z500[valid_dates_ind,ens_sel-1,:,:].values.reshape(len(valid_dates_ind)*s2,s3,s4)
    Z500_mean = z500[valid_dates_ind,ens_sel-1,:,:].mean(axis=0)
    #Z500_anoms = Z500_mean - z500[valid_dates_summer,ens_sel-1,:,:].mean(axis=0)
    Z500_anoms = Z500_mean - z500[valid_dates_summer,ens_sel-1,:,:]
            #
        #    fig1 = plt.figure()                
        #    lons, lats = np.meshgrid(lon,lat)
        #    ax1 = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
        #    ax1.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
        #    cp1 = ax1.pcolormesh(lons,lats,Z500_anoms*0.01,transform=ccrs.PlateCarree(),
        #                         cmap='RdBu_r',vmin=-10,vmax=10)
        #    cbar = plt.colorbar(cp1,pad=0.1)
        #    cbar.ax.set_xlabel('m^2-s^-2')
        #    gl = ax1.gridlines()
        #    gl.nsteps = 90
        #    ax1.text(-1,20,'0',fontsize=10,transform=ccrs.PlateCarree())
        #    ax1.text(-179,28.5,'180',fontsize=10,transform=ccrs.PlateCarree())
        #    ax1.text(119,16.5,'120E',fontsize=10,transform=ccrs.PlateCarree())
        #    ax1.text(-60,6.5,'60W',fontsize=10,transform=ccrs.PlateCarree())
        #    ax1.text(-40,20.5,'20N',fontsize=10,transform=ccrs.PlateCarree())
        #    ax1.coastlines()
        #    title_str = 'Z500 anomalies on VRILE days, {region}, ensemble {ens_no}'.format(region=region_sel_test,ens_no=ens_sel)
        #    ax1.set_title(title_str,fontsize=12)
        #    plt.show()
        #    save_dir = '/home/disk/sipn/mcmcgraw/figures/VRILE/atmosphere/{model_name}/{model_type}/{region}/'.format(model_name=model_name,
        #                                                                  model_type=model_type,region=region_sel_test)
        #    if os.path.exists(save_dir) == False:
        #        os.mkdir(save_dir)
        #        fname_save = save_dir+'Z500_anoms_VRILE_days_ens{ens_no}_{region}.png'.format(ens_no=iens+1,
        #                                               region=region_sel_test)
        #        fig1.savefig(fname_save,format='png',dpi=600)
        #    else:
        #        fname_save = save_dir+'Z500_anoms_VRILE_days_ens{ens_no}_{region}.png'.format(ens_no=iens+1,
        #                                               region=region_sel_test)
        #        fig1.savefig(fname_save,format='png',dpi=600)
    if iens == 0:
        Z500_anoms_ALL_ENS = Z500_anoms
#    elif iens == 1:
#        Z500_anoms_ALL_ENS = np.stack((Z500_anoms_ALL_ENS,Z500_anoms),axis=2)
    else:
        Z500_anoms_ALL_ENS = np.dstack((Z500_anoms_ALL_ENS,Z500_anoms))

    #plt.close('all')
Z500_mean = np.nanmean(Z500_anoms_ALL_ENS,axis=2)
Z500_sign = np.sign(Z500_anoms_ALL_ENS)
Z500_count = np.count_nonzero(Z500_sign == 1, axis=2)
    