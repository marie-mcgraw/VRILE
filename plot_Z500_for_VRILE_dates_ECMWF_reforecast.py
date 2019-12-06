#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:17:25 2019

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
ireg = 'Kara-Laptev'
print('now plotting {region}'.format(region=ireg))
region_sel_test = ireg
SIE_region_sel_test = SIE_region_groups.get_group(region_sel_test)
#for iens in np.arange(0,no_ens):
iens = 0
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
#VRILE_dates_Z500_mons = np.array([])
Z500_mean_mons = np.array([])
Z500_anoms_mons = np.array([])
for imon in np.arange(0,len(mon_sel_ind)):
    #all VRILE dates in that month
    VRILE_dates_Z500 = ((valid_dates.isin(VRILE_dates)) & (valid_dates.month== mon_sel_ind[imon]))
    #all days in that month
    valid_dates_summer = valid_dates.month == mon_sel_ind[imon]
    valid_dates_summer = valid_dates_summer.flatten()
    #all VRILE dates in that month
    valid_dates_ind = np.asarray(np.where(VRILE_dates_Z500 == True))
    valid_dates_ind = valid_dates_ind.flatten()
    
    #Z500_select = z500[valid_dates_ind,ens_sel-1,:,:].values.reshape(len(valid_dates_ind)*s2,s3,s4)
    Z500_mean = z500[valid_dates_ind,ens_sel-1,:,:].mean(axis=0)
    Z500_anoms = Z500_mean - z500[valid_dates_summer,ens_sel-1,:,:].mean(axis=0)
    if imon == 0:
        Z500_mean_mons = Z500_mean
        Z500_anoms_mons = Z500_anoms
    else:
        Z500_mean_mons = np.dstack((Z500_mean_mons,Z500_mean))
        Z500_anoms_mons = np.dstack((Z500_anoms_mons,Z500_anoms))
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
    Z500_anoms_ALL_ENS = Z500_anoms_mons
elif iens == 1:
    Z500_anoms_ALL_ENS = np.stack((Z500_anoms_ALL_ENS,Z500_anoms_mons),axis=2)
else:
    Z500_anoms_ALL_ENS = np.dstack((Z500_anoms_ALL_ENS,Z500_anoms_mons))

    #plt.close('all')
Z500_mean = np.nanmean(Z500_anoms_ALL_ENS,axis=2)
Z500_sign = np.sign(Z500_anoms_ALL_ENS)
Z500_count = np.count_nonzero(Z500_sign == 1, axis=2) #number of ensembles where Z500 anoms are positive
#Figure
Z500_all_SD = z500[valid_dates_summer,ens_sel-1,:,:].std(axis=0)
Z500_VRILE_SD = z500[valid_dates_ind,ens_sel-1,:,:].std(axis=0)
t_denom = ((Z500_all_SD**2)/len(valid_dates_summer)) + ((Z500_VRILE_SD**2)/len(valid_dates_ind))
t = Z500_mean/np.sqrt(t_denom)
tscore = t.values
N1 = len(valid_dates_ind)
N2 = len(valid_dates_summer)
dof = min(N1,N2)
#pvals = 1 - stats.t.cdf(tscore,df=dof)
alpha = 0.05
alpha_twotail = alpha/2
t_crit = stats.t.ppf(alpha_twotail,dof)
fig2 = plt.figure()
lons, lats = np.meshgrid(lon,lat)
ax2 = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
ax2.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
cp2 = ax2.pcolormesh(lons,lats,Z500_mean_mons[:,:,0],transform=ccrs.PlateCarree(),
                     cmap='RdBu_r')#,vmin=-100,vmax=100)
#cp2 = ax2.pcolormesh(lons,lats,Z500_count,transform=ccrs.PlateCarree(),
#                     cmap='RdBu_r',vmin=0,vmax=25)
cbar2 = plt.colorbar(cp2,pad=0.1)
#cbar2.ax.set_yticklabels(['25','20','15','15','20','25'])
ax2.coastlines()
#ax2.gridlines()
title_str = 'Agreement on sign of Z500 anomalies on VRILE days, all ensembles, {region} ({mon_str})'.format(region=region_sel_test,mon_str=mon_str)
ax2.set_title(title_str,fontsize=12)
#plt.show()
fname2 = "/home/disk/sipn/mcmcgraw/figures/VRILE/atmosphere/{model_name}/{model_type}/"\
"{region}_{mon_str}_Z500_anoms_sign_agreement_VRILE_days.png".format(model_name=model_name,
 model_type='reforecast',region=region_sel_test,mon_str=mon_str)
fig2.savefig(fname2,format='png',dpi=600,bbox_inches='tight')
#Same as figure 2 but only showing significant values
positive_count = count_sig
negative_count = no_ens-count_sig
Z500_count_filt = Z500_count
Z500_count_filt = Z500_count_filt.astype('float')
Z500_count_filt[(Z500_count_filt <= positive_count) & (Z500_count_filt >= negative_count)] = np.nan
#Z500_count_filt = Z500_count_filt.astype('float')
#Z500_count_filt[Z500_count_filt==0] = np.nan
fig5 = plt.figure()
lons, lats = np.meshgrid(lon,lat)
ax5 = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
ax5.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
cp5 = ax5.pcolormesh(lons,lats,Z500_count_filt,transform=ccrs.PlateCarree(),
                     cmap='RdBu_r',vmin=0,vmax=25)
cbar5 = plt.colorbar(cp5,pad=0.1)
cbar5.ax.set_yticklabels(['25','20','15','15','20','25'])
ax5.coastlines()
#ax2.gridlines()
title_str5 = 'Agreement on sign of Z500 anomalies on VRILE days, all ensembles, {region} ({mon_str})'.format(region=region_sel_test,mon_str=mon_str)
ax5.set_title(title_str5,fontsize=12)
fname5 = "/home/disk/sipn/mcmcgraw/figures/VRILE/atmosphere/{model_name}/{model_type}/"\
"{region}_{mon_str}_Z500_anoms_sign_agreement_VRILE_days_ONLY_SIGNIFICANT.png".format(model_name=model_name,
 model_type='reforecast',region=region_sel_test,mon_str=mon_str)
fig5.savefig(fname5,format='png',dpi=600,bbox_inches='tight')
#
fig3 = plt.figure()
lons, lats = np.meshgrid(lon,lat)
ax3 = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
ax3.set_extent([-180, 180, 25, 90], crs=ccrs.PlateCarree())
#max_cb = round(np.amax(abs(Z500_mean)/100)/5)*5
cp3 = ax3.pcolormesh(lons,lats,0.0981*Z500_mean,transform=ccrs.PlateCarree(),
                     cmap='RdBu_r')#,vmin=-max_cb,vmax=max_cb)
cbar3 = plt.colorbar(cp3,pad=0.1)
cbar3.ax.set_xlabel('m^2-s^-2')
ax3.coastlines()
title_str3 = 'Ensemble Mean Z500 anomalies on VRILE days, {region} ({mon_str})'.format(region=region_sel_test,mon_str=mon_str)
ax3.set_title(title_str3,fontsize=12)
#plt.show()    
fname3 = "/home/disk/sipn/mcmcgraw/figures/VRILE/atmosphere/{model_name}/{model_type}/"\
"{region}_{mon_str}_Z500_anoms_ens_mean_VRILE_days.png".format(model_name=model_name,
 model_type='reforecast',region=region_sel_test,mon_str=mon_str)
fig3.savefig(fname3,format='png',dpi=600,bbox_inches='tight') 
#
fig4 = plt.figure()
ax4 = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
ax4.set_extent([-180,180,25,90], crs=ccrs.PlateCarree())
max_cb4 = round(np.amax(abs(Z500_mean)/100)/5)*5
Z500_filt = 0.01*Z500_mean
Z500_filt[abs(tscore) < abs(t_crit)] = np.nan
cp4 = ax4.pcolormesh(lons,lats,Z500_filt,transform=ccrs.PlateCarree(),
                     cmap='RdBu_r',vmin=-max_cb,vmax=max_cb)
cbar4 = plt.colorbar(cp4,pad=0.1)
cbar4.ax.set_xlabel('m^2-s^-2')
pl4 = ax4.plot()
ax4.coastlines()
title_str4 = 'Ensemble Mean Z500 anomalies on VRILE days, {region} ({mon_str})'.format(region=region_sel_test,mon_str=mon_str)
ax4.set_title(title_str4,fontsize=12)
fname4 = "/home/disk/sipn/mcmcgraw/figures/VRILE/atmosphere/{model_name}/{model_type}/"\
"{region}_{mon_str}_Z500_anoms_ens_mean_VRILE_days_ONLY_SIGNIFICANT.png".format(model_name=model_name,
 model_type='reforecast',region=region_sel_test,mon_str=mon_str)
fig4.savefig(fname4,format='png',dpi=600,bbox_inches='tight') 

plt.close('all')