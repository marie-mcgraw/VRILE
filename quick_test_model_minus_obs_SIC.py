#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:18:14 2019

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


model_name = 'ecmwfsipn'
model_type = 'reforecast'
filepath = '/home/disk/sipn/nicway/data/model/{model_name}/{model_type}/sipn_nc/'.format(model_name=model_name,
                                              model_type=model_type)
filenames = xr.open_mfdataset(filepath+'/*.nc',concat_dim='init_time')
print(filenames)

# In[39]:

date_choose = '1999-01-01'
yr_sel = '1999'
#region_names = filenames.region_names
#region_names = np.append(region_names,['Kara-Laptev','East-Siberian-Beaufort-Chukchi'])
init_times = filenames.init_time
forecast_times = filenames.fore_time
init_time_trim = init_times.where((init_times.dt.year==2014)&(init_times.dt.month==8)&(init_times.dt.day==1))
sic = filenames.sic
sic_trim = sic.sel(init_time=date_choose)

#%%
#Pick lead day 2, take ensemble mean
sic_lead = sic_trim.isel(fore_time=0)
sic_ens_mean = sic_lead.mean('ensemble')
##chunk sizes in dimensions of [init_time x ensemble x fore_time x region]
#chunk_sizes = filenames['Extent'].shape
#%%
# plot
fig1 = plt.figure(1)
ax1 = fig1.add_axes([0.1,0.1,0.9,0.9])
sic_ens_mean.plot(ax=ax1)
ax1.set_title('{model_name}, lead: 0, {date_choose}'.format(model_name=model_name,
              date_choose=date_choose))
fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE_v2/sea_ice_concentration/'
fname_1 = fpath_save+'{model_name}_{model_type}_{date_choose}_SIC_lead_0.png'.format(model_name=model_name,
           model_type=model_type,date_choose=date_choose)
fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
#%%OBS
obs_name = 'NSIDC_0079'
filepath_obs = '/home/disk/sipn/nicway/data/obs/{model_name}/sipn_nc_yearly/'.format(model_name=obs_name)
filenames_obs = xr.open_dataset(filepath_obs+'/{yr_sel}.nc'.format(yr_sel=yr_sel))#,concat_dim='time')

sic_obs = filenames_obs.sic

#%%
sic_obs_trim = sic_obs.sel(time=date_choose).transpose()
fig2 = plt.figure(2)
ax2 = fig2.add_axes([0.1,0.1,0.9,0.9])
sic_obs_trim.plot(ax=ax2)
ax2.set_title('{obs_name}, {date_choose}'.format(obs_name=obs_name,date_choose=date_choose))
fname_2 = fpath_save+'{model_name}_{date_choose}_SIC_lead_0.png'.format(model_name=obs_name,
           date_choose=date_choose)
fig2.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')
#%%
#sic_obs_trans = sic_obs_trim.transpose()
diff = sic_ens_mean.values - sic_obs_trim.values

#%%
fig3 = plt.figure(3)
ax3=fig3.add_axes([0.1,0.1,0.9,0.9])
cp3 = ax3.contourf(diff,cmap='seismic',levels=9,vmin=-1,vmax=1.01,extend='both')
cbar3 = fig3.colorbar(cp3,ax=ax3)
ax3.set_title('{model_name} minus {obs_name}, {date_choose}'.format(model_name=model_name,
              obs_name=obs_name,date_choose=date_choose))
fname_3 = fpath_save+'{model_name}_{model_type}_minus_{obs_name}_{date_choose}_SIC_lead_0.png'.format(model_name=model_name,
           model_type=model_type,obs_name=obs_name,date_choose=date_choose)
fig3.savefig(fname_3,format='png',dpi=600,bbox_inches='tight')