#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import scipy.stats as stats
import os.path


# Define a plotting function for polar stereographic plots.  X is field to plot, max_lat is how far out the plot should zoom, 
# #cbar_lims is the limit for the colorbar

# In[2]:


def polar_stereo_plot_anom(lon,lat,X,max_lat,cbar_lims):
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax.set_extent([-180,180,max_lat,90],crs=ccrs.PlateCarree())
    cp = ax.pcolormesh(lon,lat,X,transform=ccrs.PlateCarree(),
                       cmap='RdBu_r',vmin=-cbar_lims,vmax=cbar_lims)
    cbar = plt.colorbar(cp,pad=0.05,extend='both')
    ax.coastlines()
    return ax, cbar, fig


# This function calculates the difference of means for our composites

# In[3]:


def diff_of_means(X1,X2,alpha,N1,N2,select_axis):
    sigma_1 = X1.std(axis=select_axis)
    sigma_2 = X2.std(axis=select_axis)
    mu_1 = X1.mean(axis=select_axis)
    mu_2 = X2.mean(axis=select_axis)
    t_denom = (sigma_1**2)/N1 + (sigma_2**2)/N2
    tscore = (mu_2 - mu_1)/np.sqrt(t_denom)
    dof = min(N1,N1)
    tcrit = stats.t.ppf(alpha,dof)
    return tscore, tcrit


# We'll load the ECMWF Z500 data with xarray

# In[4]:



model_name = 'ecmwfc3s'
#model_type = 'reforecast'
variable_name = 'Z500'
filepath = '/home/disk/sipn/mcmcgraw/data/atmospheric_data/{model_name}/{variable_name}/'.format(model_name=model_name,
                                                           variable_name=variable_name)

filename = xr.open_mfdataset(filepath+'/*.nc',concat_dim='time')
lon = filename.longitude
lat = filename.latitude
z500 = filename.z
z500_date = filename.time
z500_date = pd.DatetimeIndex(z500_date.values)
no_ens = len(z500['number'])


# Load SIE data so we can define VRILEs

# In[5]:


model_name_SIE = 'ecmwfsipn'
model_type_SIE = 'reforecast'
no_day_change = 5 #looking at 5 day changes
day_change = no_day_change
max_lead = 30
delta_lead_days = np.arange(no_day_change,max_lead+1,1)
delta_first_days = np.arange(1,max_lead+2-no_day_change,1)
no_forecast_periods = len(delta_lead_days)
SIE_filepath_load = '/home/disk/sipn/mcmcgraw/data/VRILE/'
SIE_filename_full = SIE_filepath_load+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name_SIE,
                               model_type=model_type_SIE,d_days=no_day_change)

SIE_file_full = pd.read_csv(SIE_filename_full)
SIE_file_full = SIE_file_full.dropna()


# Pick a region to test for sea ice.  We'll always look at the full field of Z500 

# In[6]:


regions_list = SIE_file_full['region'].unique().tolist()
#for ireg in np.arange(12,len(regions_list)):
region_choose_ind = 7
region_sel = regions_list[region_choose_ind]
print('selecting region {region}'.format(region=region_sel))
SIE_group_region = SIE_file_full.groupby(['region'])
SIE_choose_region = SIE_group_region.get_group(region_sel)


# Select an ensemble that we're interested in.  We'll do each ensemble separately since Z500 and sea ice ensembles should correspond

# In[7]:

for iens in np.arange(0,no_ens):
#iens= 0
    ensemble_choose = iens+1
    ensemble_ind = iens
    SIE_group_ens = SIE_choose_region.groupby(['ensemble'])
    SIE_choose_ens = SIE_group_ens.get_group(ensemble_choose)
    z500_ens = z500.sel(number=ensemble_ind)
    
    
    # Pick the months that we're interested in--JJAS.  For the VRILE, we'll define our 5th percentile events using all 4 months
    
    # In[8]:
    
    
    d_SIE_select = SIE_choose_ens['d_SIE (V - I)']
    full_seas = [6,7,8,9]
    seas_str = 'JJAS'
    SIE_valid = SIE_choose_ens['V (valid date)']
    SIE_valid_df = pd.DatetimeIndex(SIE_valid)
    full_seas_ind = np.isin(SIE_valid_df.month,full_seas)
    SIE_valid_dates_seas = SIE_valid.where(full_seas_ind==True)
    d_SIE_seas = d_SIE_select.where(full_seas_ind==True).dropna()
    d_SIE_seas_p5 = d_SIE_seas.quantile(0.05)
    
    
    # For Z500, we'll do each month separately.  First, remove annual mean.  Then slim down to only the months we care about and then remove the mean Z500 from each month separately. 
    
    # In[9]:
    
    
    z500_time_months = z500['time.month']
    z500_ens = z500_ens - z500_ens.mean('time')
    z500_months_ind = np.isin(z500_time_months,full_seas)
    z500_month_trim = z500_ens[z500_months_ind,:,:]
    z500_months_groups = z500_month_trim.groupby('time.month')
    z500_months_anoms = z500_months_groups - z500_months_groups.mean('time')
    #z500_months_anoms_groups = z500_months_anoms.groupby('time.month')
    #z500_months_anoms.mean(dim=('longitude','latitude')).plot(marker='o')
    lon = z500_ens['longitude']
    lat = z500_ens['latitude']
    lons, lats = np.meshgrid(lon,lat)
    #
    
    
    # Quick sanity check plot.  If we do a month's anomalies, it should look like noise. If we plot all Augusts with annual mean removed, we should see high anomalies everywhere. If we plot all Januarys with the annual mean removed, we should see low anomalies everywhere. 
    
    # In[10]:
    
    
    #foo = np.where(z500_ens['time.month']==8)
    #zplt = z500_ens.isel(time=foo[0]).mean('time')
    #zplt[0:10,0:10].values
    #z500_months_anoms.isel(time=foo[0]).mean('time').values
    #ax1 = polar_stereo_plot_anom(lons,lats,0.0981*zplt,55,600)
    
    
    # Now, find the Z500 anomalies that correspond to the VRILE days
    
    # In[11]:
    
    
    d_SIE_seas_VRILE = d_SIE_seas.where(d_SIE_seas <= d_SIE_seas_p5).dropna()
    dates_SIE_seas_VRILE = SIE_valid_dates_seas.where(d_SIE_seas <= d_SIE_seas_p5).dropna()
    z500_dates_ind = z500_months_anoms['time'].isin(dates_SIE_seas_VRILE)
    z500_dates_VRILE = z500_months_anoms.isel(time=z500_dates_ind)
    
    z500_dates = pd.to_datetime(z500_months_anoms['time'].values)
    z500_dates_VRILE = z500_dates.isin(dates_SIE_seas_VRILE)
    z500_VRILE = z500_months_anoms[z500_dates_VRILE]
    
    # concatenate along number dimension
    if iens == 0:
        z500_VRILE_all_ens = z500_VRILE
    else:
        z500_VRILE_all_ens = xr.concat([z500_VRILE_all_ens,z500_VRILE],dim='number')
    # Plot full field
    
    # In[31]:
    var_name = 'z500'
    
    ax1,cbar1,fig1 = polar_stereo_plot_anom(lons,lats,(1/9.81)*z500_VRILE.mean('time'),55,20)
    ax1.set_title('Z500 anomalies on VRILE days, {region}, {month}, ensemble {ens}'.format(region=region_sel,
              month=seas_str,ens=iens+1))
    cbar1.ax.set_xlabel('10^1 m')
    fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE_v2/atmosphere/{var_name}/VRILE_composites/'.format(var_name=var_name)
    
    fname_1 = fpath_save+'all_points/{model_name}_z500_anomalies_VRILE_days_{region}_{seas}_ens{ens}.png'.format(model_name=model_name,
                                     region=region_sel,seas=seas_str,ens=iens+1)
    fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
    # Now, plot only significant data points
    
    # In[32]:
    
    
    N1 = len(z500_months_anoms['time'])
    N2 = len(z500_VRILE['time'])
    alpha = 0.05
    
    tscore, tcrit = diff_of_means(z500_months_anoms,z500_VRILE,alpha,N1,N2,0)
    tscore_plot = tscore.where(abs(tscore) > abs(tcrit))
    z500_plot = z500_VRILE.mean('time').where(abs(tscore) > abs(tcrit))
    ax2,cbar2,fig2 = polar_stereo_plot_anom(lons,lats,0.1*z500_plot,55,20)
    ax2.set_title('Z500 anomalies on VRILE days, {region}, {month}, ensemble {ens}'.format(region=region_sel,
              month=seas_str,ens=iens+1))
    cbar2.ax.set_xlabel('10^1 m')
    fname_2 = fpath_save+'sig_points_only/{model_name}_z500_anomalies_VRILE_days_{region}_{seas}_ens{ens}.png'.format(model_name=model_name,
            region=region_sel,seas=seas_str,ens=iens+1)
    fig2.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')
    
    plt.close('all')
# In[ ]:


#z500_ensemble_mean = z500_VRILE_all_ens.mean('number')
z500_VRILE_days = z500_VRILE_all_ens
file_name_write = '/home/disk/sipn/mcmcgraw/data/VRILE/{model_name}_Z500_anomalies_VRILE_days_in_{region}_{seas}.nc'.format(model_name=model_name,
                          region=region_sel,seas=seas_str)
if os.path.exists(file_name_write)==False:
    z500_VRILE_days.to_netcdf(file_name_write)

z500_VRILE_days.close()
z500_VRILE_all_ens.close()

