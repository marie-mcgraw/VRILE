#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import scipy.stats as stats


#define plotting function. X is field to plot, max_lat is how far out the plot should zoom, 
#cbar_lims is the limit for the colorbar
def polar_stereo_plot_anom(lon,lat,X,max_lat,cbar_lims):
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax.set_extent([-180,180,max_lat,90],crs=ccrs.PlateCarree())
    cp = ax.pcolormesh(lon,lat,X,transform=ccrs.PlateCarree(),
                       cmap='RdBu_r',vmin=-cbar_lims,vmax=cbar_lims)
    cbar = plt.colorbar(cp,pad=0.05,extend='both')
    ax.coastlines()
    return ax

#define difference of means
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
# Load Z500 data

# In[15]:


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
no_ens = 25

# Now load SIE data

# In[217]:


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
# Now pick a region for VRILEs

# In[219]:


regions_list = SIE_file_full['region'].unique().tolist()
region_choose_ind = 16
region_sel = regions_list[region_choose_ind]
print('selecting region {region}'.format(region=region_sel))
SIE_group_region = SIE_file_full.groupby(['region'])
SIE_choose_region = SIE_group_region.get_group(region_sel)

# Filter SIE dataframe by ensemble

# In[218]:

#for iens in np.arange(0,5):
iens= 0
ensemble_choose = iens+1
ensemble_ind = iens
SIE_group_ens = SIE_choose_region.groupby(['ensemble'])
SIE_choose_ens = SIE_group_ens.get_group(ensemble_choose)

# Select the 5-day change in sea ice (VRILE) and identify 5th percentile events relative to our full season. Since our SIE time series is shorter than Z500, don't include any dates that aren't in SIE

# In[220]:


d_SIE_select = SIE_choose_ens['d_SIE (V - I)']
full_seas = [6,7,8,9]
seas_str = 'JJAS'
SIE_valid = SIE_choose_ens['V (valid date)']
SIE_valid_df = pd.DatetimeIndex(SIE_valid)
full_seas_ind = np.isin(SIE_valid_df.month,full_seas)
SIE_valid_dates_seas = SIE_valid.where(full_seas_ind==True)
d_SIE_seas = d_SIE_select.where(full_seas_ind==True).dropna()
d_SIE_seas_p5 = d_SIE_seas.quantile(0.05)


# Create a histogram of JJAS d_SIE_seas. Red line indicates 5th percentile 

# In[221]:

#    fig2 = plt.figure()
#    ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
#    ax2 = d_SIE_seas.hist()
#    ax2.axvline(x=d_SIE_seas_p5,color='r',linewidth=2)
#    plt.ylabel('count')
#    plt.xlabel('5-day change in SIE')
#    plt.title('5-day change in SIE for {seas}, ensemble {ens_no}, region {region_name}'.format(seas=seas_str,
#                                                                                              ens_no=ensemble_choose,
#                                                                                              region_name=region_sel))
# Choose a single ensemble member and a month to analyze, then select appropriate Z500

# In[214]:

z500_ens_sel = z500[:,ensemble_ind,:,:]
z500_annual_mean_rem = z500_ens_sel - z500_ens_sel.mean(axis=0)
z500_VRILE_all_seas = np.array([])
d_Z500_VRILE_all_seas = np.array([])
for imon_sel in full_seas:
    month_sel = imon_sel
    date_sel_mon_ind = np.where(z500_date.month == month_sel)
    date_mon_choose = z500_date.where(z500_date.month == month_sel).dropna()
    z500_mon_choose = z500_annual_mean_rem[date_sel_mon_ind[0],:,:]
    
    
    # Plot what average z500 looks like in chosen month (high anoms everywhere in august, low everywhere in jan, so that's good)
    
    # In[216]:
    
#        ax1 = polar_stereo_plot_anom(lons,lats,0.0981*z500_mon_choose.mean(axis=0),55,60)
#        ax1.set_title('mean Z500 anomalies in month {mon} for ensemble {ens_no}'.format(mon=month_sel,ens_no=ensemble_choose))
#        
    
    
    # Now, create Z500 anomalies for each month separately, based on VRILE.  First identify the VRILEs in the SIE data
    
    # In[222]:
    
    
    one_month_ind = np.isin(SIE_valid_df.month,month_sel)
    d_SIE_one_month = d_SIE_select.where(one_month_ind==True).dropna()
    one_month_VRILE_thresh = d_SIE_one_month.quantile(0.05)
    d_SIE_one_month_VRILE = d_SIE_one_month.where(d_SIE_one_month <= one_month_VRILE_thresh).dropna()
    
    
    # Use the SIE anomalies to find the Z500 anomalies
    
    # In[223]:
    
    
    SIE_valid_Z500_choose = SIE_valid_df.where(one_month_ind==True).dropna()
    z500_date_trim_ind = date_mon_choose.isin(SIE_valid_Z500_choose)
    z500_date_trim_ind = date_mon_choose.isin(SIE_valid_Z500_choose)
    z500_mon_sel = z500_mon_choose[z500_date_trim_ind.flatten(),:,:]
    z500_VRILE_ind = np.where(d_SIE_one_month <= one_month_VRILE_thresh)
    z500_VRILE = z500_mon_sel[z500_VRILE_ind[0],:,:]
    
    # Now make the z500 composites
    
    # In[224]:
    
#        ax3 = polar_stereo_plot_anom(lons,lats,0.0981*z500_VRILE.mean(axis=0),55,60)
#        ax3.set_title('mean Z500 anomalies on VRILE days in month {mon} for ensemble {ens_no}'.format(mon=month_sel,ens_no=ensemble_choose))
#        
    
    # In[225]: plot d_Z500 for VRILES
    
    
    d_z500_VRILE = z500_VRILE.mean(axis=0) - z500_mon_sel.mean(axis=0)
    alpha = 0.05
    alpha_twotail = alpha/2
    N1 = len(z500_VRILE_ind[0])
    N2 = len(z500_date_trim_ind.flatten())
    axis_select = 0
    tscore_seas,tcrit_seas = diff_of_means(z500_VRILE,z500_mon_sel,alpha_twotail,
                                           N1,N2,axis_select)
    t_mask = abs(tscore_seas).values
    t_mask[t_mask <= abs(tcrit_seas)] = np.nan
    t_mask[t_mask >= abs(tcrit_seas)] = 1
    
     # Create array for all 4 months
    if month_sel == full_seas[0]:
        z500_all_seas = z500_mon_sel.mean(axis=0)
        z500_sigma_all_seas = z500_mon_sel.std(axis=0)
        z500_VRILE_all_seas = z500_VRILE.mean(axis=0)
        z500_sigma_VRILE_all_seas = z500_VRILE.std(axis=0)
        d_z500_VRILE_all_seas = d_z500_VRILE
        t_mask_all_seas = t_mask
    else:
        z500_all_seas = np.dstack((z500_all_seas,z500_mon_sel.mean(axis=0)))
        z500_sigma_all_seas = np.dstack((z500_sigma_all_seas,z500_mon_sel.std(axis=0)))
        z500_VRILE_all_seas = np.dstack((z500_VRILE_all_seas,z500_VRILE.mean(axis=0)))
        z500_sigma_VRILE_all_seas = np.dstack((z500_sigma_VRILE_all_seas,z500_VRILE.std(axis=0)))
        d_z500_VRILE_all_seas = np.dstack((d_z500_VRILE_all_seas,d_z500_VRILE))
        t_mask_all_seas = np.dstack((t_mask_all_seas,t_mask))
        
#        ax4 = polar_stereo_plot_anom(lons,lats,0.0981*d_z500_VRILE,55,100)
#        ax4.set_title('Z500_VRILE - Z500_ALL, month {mon} for ensemble {ens_no}'.format(mon=month_sel,ens_no=ensemble_choose))
#    
    plt.close('all')

# Full script analysis:
#     Loop through region
#     
#     
#     Calculate difference of means for each Z500 anomaly composite
#     Take full composite: 2 significance tests
#         must be significant in EACH month
#         significant in ANY month
#     Sign agreement from all 4 smushed together
#     Field significance test

# In[ ]:
# Determine significance from diff of means.  3 ways:
    #1 if at least 2 months have non-NaN at that gridpoint, significant
    #2 if ANY months have non-NaN at that gridpoint, significant
    #3 take average of z500 and do difference of means on that 
t_count = np.count_nonzero(~np.isnan(t_mask_all_seas),axis=2)
t_mask_V1 = np.ones(t_count.shape)
t_mask_V1[t_count < 3] = np.nan
t_mask_V2 = np.ones(t_count.shape)
t_mask_V2[t_count < 1] = np.nan 
N1_full_seas = len(full_seas_ind)
N2_full_seas = 0.05*N1_full_seas
#tdenom_seas_mean = 
# Plot composite for entire season
d_z500_VRILE_all_seas_mean = d_z500_VRILE_all_seas.mean(axis=2)
ax5 = polar_stereo_plot_anom(lons,lats,0.0981*d_z500_VRILE_all_seas.mean(axis=2),55,60)
ax5.set_title('Z500_VRILE - Z500_ALL, seas {mon} for ensemble {ens_no}, region {region}'.format(mon=seas_str,ens_no=ensemble_choose,region=region_sel))

plt.close('all')
#Now stack for ensemble members
#if iens == 0:
#    d_z500_VRILE_all_ens = d_z500_VRILE_all_seas_mean
#else:
#    d_z500_VRILE_all_ens = np.dstack((d_z500_VRILE_all_ens,d_z500_VRILE_all_seas_mean))
#    
# plot all
ax7 = polar_stereo_plot_anom(lons,lats,0.0981*d_z500_VRILE_all_ens.mean(axis=2),55,60)
ax7.set_title('Z500_VRILE - Z500_ALL, seas {mon} for all ensembles, region {region}'.format(mon=seas_str,ens_no=ensemble_choose,region=region_sel))
