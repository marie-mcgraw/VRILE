#!/usr/bin/env python
# coding: utf-8

# <code>calculate_RMSE_MAE_damped_anomaly_forecast.ipynb</code>. This notebook outputs RMSE and MAE for the damped anomaly forecasts of SIE and anomalous SIE. Model forecasts are separated into VRILE and non-VRILE days.

# ### Load packages

# In[1]:


import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut


# In[4]:


def create_aggregate_regions_DAMPED_ANOM(SIE_data):
    regions_agg_list = [['Kara Sea','Laptev Sea'],['Barents Sea','Kara Sea','Laptev Sea'],['East Siberian Sea','Beaufort Sea','Chukchi Sea'],
                       ['Baffin Bay','East Greenland Sea'],['East Siberian Sea','Beaufort Sea','Chukchi Sea','Laptev Sea']]
    region_names_extra = ['Kara-Laptev Sea','Barents-Kara-Laptev Sea','East Siberian-Beaufort-Chukchi Sea',
                      'Atlantic','East Siberian-Beaufort-Chukchi-Laptev Sea']
    #
    for i_reg in np.arange(0,len(regions_agg_list)):
        i_reg_sel = regions_agg_list[i_reg]
        SIE_ireg = SIE_data[SIE_data['region'].isin(i_reg_sel)]
        SIE_ireg_group = SIE_ireg.set_index(['region','init date','valid date','lead time']).mean(level=(1,2,3))
        SIE_ireg_group['region'] = region_names_extra[i_reg]
        SIE_data =SIE_data.append(SIE_ireg_group.reset_index())
        
    return(SIE_data)


# ### Overview
# Okay. Overview of our cross-validated, significance-tested VRILE error. 
# * Load damped anomaly netCDF files and use common reforecast period.  Use data that has been regridded to the common land mask. 
# * Add aggregate regions 
# * Create climatology--damped anomaly model: calculate date of year for valid date, lead time in weeks.
#     *  Group by region, lead time, and valid date of year 
#     *  Average climatology based on day of year and lead time in weeks--use <code>transform</code> to create <code>SIE_clim</code>.
#     *  Subtract <code>SIE_clim</code> from <code>SIE</code>
# * Create observed climatology based on common reforecast period (1999-2014)
# * Set up LeaveOneOut cross-validation: We remove each year (1999-2014) from the observations.  Then we:
#     * Calculate VRILEs excluding each year
#     * Identify forecasts that correspond to VRILE days and separate S2S model data into VRILE days and non-VRILE days
#     * Calculate errors: as a function of region, valid date, and lead time. 
#     * Assess significance:
#         * H0: RMSE for non-VRILE days = RMSE for VRILE days 
#         * Calculate p-value: $p_0 = \frac{(RMSE_{VRILE} - RMSE_{NOVRILE})}{\sqrt{\frac{S_{VRILE}^2}{N_{VRILE}} + \frac{S_{NOVRILE}^2}{N_{NOVRILE}}}}$
#         * Save p-values, standard deviations as a function of lead time, region, and year left out. 
#         * When $|p| > |p_{crit}|$ ($p_{crit} = \pm 1.96$), we can say that the model's ability to predict sea ice on VRILE days is significantly different from the model's ability to predict sea ice on non-VRILE days in that region for that lead time while leaving out that year
#   
# * How many years must be significantly different for us to say our samples are overall different? Use a binomial test
#     * $\sum_{i}^{N} {N \choose i}p^i(1 - p)^{N - i}$
#     * N: total number of samples (15, one for each year between 1999-2014)
#     * p: 0.5 (assume we have equal probability of rejecting or not-rejecting null hypothesis)
#     * we need to find i: i = 13 for rejecting hypothesis at 95% confidence

# In[2]:


from S2S_sea_ice_preprocess import load_model, create_aggregate_regions, create_model_climatology, create_obs_climatology
from S2S_sea_ice_VRILEs import get_VRILE_days
from S2S_sea_ice_metrics import calculate_errors, get_pvalues


# <b>inputs:</b><br>
# <li>  <code>model name</code>: (ecmwf,ukmo,ncep,metreofr) </li>
# <li>  <code>seas_str</code>: [string for season; ALL if we want to do full year]</li>
# <li>  <code>seas_sel</code>: [months of season; empty if we want to do full year] </li>
# <li>  <code>vrile_thresh</code>: [threshhold at which VRILE is estimated </li>
# <li>  <code>thresh_str</code>: [string for VRILE threshhold] </li>
# <li>  <code>obs_name</code>: (NSIDC_0079, NSIDC_0051, OSISAF) [observational product we are using as our "truth"]</li>
# <li>  <code>COMMON_RF</code>: boolean; indicates whether or not we want to use common reforecast period (1999-2014) or all available years (<code>True</code> is default)</li>
# <li>  <code>nday_change</code>: $n$-day change in SIE for VRILE calculation (default is 5)</li>
# <li>  <code>lead_weeks</code>: boolean; indicates whether or not we want our RMSE results to be as a function of lead days or lead weeks (default is <code>True</code>) </li>

# In[3]:


vrile_thresh = 0.05
thresh_str = '05'
nday_change = 5
seas_sel = [6,7,8]
seas_str = 'JJA'
nyear_roll = 10
lead_weeks = True
ROLL_CLIM = True


# Load

# In[ ]:


obs_name = 'NSIDC_0079'
fpath_load = '/home/disk/sipn/mcmcgraw/McGraw_etal_2020/code/make_it_nice/COMMON_LAND_MASK/data/'
fname_load = fpath_load+'{obs_name}_DAMPED_ANOMALY_FORECAST_{nyear_roll}_rolling_mean.csv'.format(obs_name=obs_name,nyear_roll=nyear_roll)
SIE_damped_a = pd.read_csv(fname_load)
SIE_damped_a = create_aggregate_regions_DAMPED_ANOM(SIE_damped_a)
SIE_damped_a['init year'] = pd.to_datetime(SIE_damped_a['init date']).dt.year
SIE_damped_a['lead time (days)'] = pd.to_timedelta(SIE_damped_a['lead time'],'D')
# Trim to common reforecast period
#SIE_damped_a = create_aggregate_regions(SIE_damped_a)
SIE_damped_a = SIE_damped_a[SIE_damped_a['init year'].isin(np.arange(1999,2015))]
SIE_damped = SIE_damped_a.set_index('region')
SIE_damped['valid date'] = pd.to_datetime(SIE_damped['valid date'])

