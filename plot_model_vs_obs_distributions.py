#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime, timedelta


# Load reforecasts (1993-2016) and forecasts (2017-2019)

# In[2]:


model_name = 'ecmwfsipn'
model_type = 'reforecast'
no_day_change = 5

filepath_save = '/home/disk/sipn/mcmcgraw/data/VRILE/'
filename_full = filepath_save+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                       model_type=model_type,d_days=no_day_change)
d_SIE = pd.read_csv(filename_full)
d_SIE_all = d_SIE.dropna()
d_SIE_all.head()

#%%
# Load the forecast model
model_type_2 = 'forecast'
filename_full_2 = filepath_save+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                       model_type=model_type_2,d_days=no_day_change)
d_SIE_forecast = pd.read_csv(filename_full_2)
# Now load observations

# In[3]:


obs_name = 'NSIDC_0079'
obs_type = 'sipn_nc_yearly_agg'
obs_filename = '/home/disk/sipn/mcmcgraw/data/VRILE/{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=obs_name,
                       model_type=obs_type,d_days=no_day_change)
SIE_obs = pd.read_csv(obs_filename)
SIE_obs = SIE_obs.dropna()


# Get lists of regions and lead days

# In[5]:


regions = d_SIE_all['region'].unique().tolist() #list unique regions
lead_days = d_SIE_all['lead time (days)'].unique().tolist() #list lead days


# Group model data by lead day and choose a lead day (here, we'll do first week, our earliest lead day)

# In[518]:


#SIE_lead_groups = d_SIE_all.groupby(['lead time (days)'])
lead_day_sel = np.arange(22,29)
lead_min = lead_day_sel[0]
SIE_lead_sel = d_SIE_all.loc[d_SIE_all['lead time (days)'].isin(lead_day_sel)]
SIE_model_dates = pd.to_datetime(SIE_lead_sel['V (valid date)'])


# Keep only observations that match dates associated with model lead times

# In[519]:


SIE_obs_dates = pd.to_datetime(SIE_obs['V (valid date)'])
f_dates = SIE_obs_dates.isin(SIE_model_dates)
SIE_obs_trim = SIE_obs.loc[f_dates]


# Select JJAS only

# In[520]:


mon_sel = [6,7,8,9]
mon_sel_str = 'JJAS'
SIE_model_JJAS = SIE_lead_sel.loc[pd.to_datetime(SIE_lead_sel['V (valid date)']).dt.month.isin(mon_sel)]
SIE_obs_JJAS_trim = SIE_obs_trim.loc[pd.to_datetime(SIE_obs_trim['V (valid date)']).dt.month.isin(mon_sel)]
SIE_obs_JJAS_all = SIE_obs.loc[pd.to_datetime(SIE_obs['V (valid date)']).dt.month.isin(mon_sel)]


# Group by region

# In[521]:


region_choose = 'East-Siberian-Beaufort-Chukchi'
region_choose_ind = 0
SIE_model_groups = SIE_model_JJAS.groupby(['region'])
SIE_obs_trim_groups = SIE_obs_JJAS_trim.groupby(['region'])
SIE_obs_all_groups = SIE_obs_JJAS_all.groupby(['region'])
#
SIE_model_sel = SIE_model_groups.get_group(region_choose)
SIE_obs_trim_sel = SIE_obs_trim_groups.get_group(region_choose)
SIE_obs_all_sel = SIE_obs_all_groups.get_group(region_choose)
#SIE_obs_trim_sel
#
#SIE_model_ens = SIE_model_sel.groupby(['ensemble'])
#SIE_ens_mean = SIE_model_ens.mean()


# Compare distributions of sea ice extent

# In[526]:


bins = np.arange(0,15,0.5)
import seaborn as sns
fig1 = plt.figure(1)
ax1 = fig1.add_axes([0.05,0.05,0.95,0.95])
ax1a = sns.kdeplot(SIE_model_sel['d_SIE (V - I)'],axes=ax1,color='xkcd:royal blue',linewidth=3)
ax1b = sns.kdeplot(SIE_obs_trim_sel['d_SIC (V - I)'],color='xkcd:brick',axes=ax1,linewidth=3)
#ax1c = sns.kdeplot(SIE_obs_all_sel['d_SIC (V - I)'],color='xkcd:brick',linestyle='--',axes=ax1,linewidth=3)
ax1.legend(('Model','Obs','obs_all'),fontsize=12)
ax1.axvline(x=SIE_model_sel['d_SIE (V - I)'].quantile(0.05),color='xkcd:royal blue',linewidth=3,linestyle=':')
ax1.axvline(x=SIE_obs_trim_sel['d_SIC (V - I)'].quantile(0.05),color='xkcd:brick',linewidth=3,linestyle=':')
ax1.set_xlabel('5-day Change in Sea Ice Extent (10$^6$ km$^2$)',fontsize=13)
ax1.set_ylabel('Density',fontsize=13)
ax1.set_title('{region}, {months}, lead days: {lead_min}-{lead_max}'.format(region=region_choose,months=mon_sel_str,
                                                                  lead_min=lead_min,lead_max=lead_day_sel[-1]),
             fontsize=15)
ax1.grid()
p05_model = SIE_model_sel['d_SIE (V - I)'].quantile(0.05).round(3)
p05_obs = SIE_obs_trim_sel['d_SIC (V - I)'].quantile(0.05).round(3)
y_loc = 7
x_loc = -2
ax1.text(p05_model*x_loc,y_loc,'model 5th pctl: {p05}'.format(p05=p05_model),fontsize=11,color='xkcd:royal blue')
ax1.text(p05_model*x_loc,y_loc-0.1*y_loc,'obs 5th pctl: {p05}'.format(p05=p05_obs),fontsize=11,color='xkcd:brick')


# In[527]:


fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE_v2/histograms/compare_obs_and_model/seas_with_p05/'
fname_save_1 = fpath_save+'{region}_{day_change}day_change_SIE_{seas}_OBS_and_model_leads_{leadmin}-{lead_max}_with_p05_{model_name}.pdf'.format(region=region_choose,
                                                                                                                 day_change=no_day_change,
                                                                                                                 seas=mon_sel_str,
                                                                                                                 leadmin=lead_min,
                                                                                                                 lead_max=lead_day_sel[-1],   
                                                                                                                 model_name=model_name)
fig1.savefig(fname_save_1,format='pdf',bbox_inches='tight')


# In[97]:


print('model ',SIE_model_sel['d_SIE (V - I)'].quantile(0.05).round(2))
print('obs trim ',SIE_obs_trim_sel['d_SIC (V - I)'].quantile(0.05))
print('obs all ',SIE_obs_all_sel['d_SIC (V - I)'].quantile(0.05))


# In[257]:


SIE_model_match_obs = SIE_model_sel.loc[SIE_model_sel['d_SIE (V - I)'] < SIE_obs_trim_sel['d_SIC (V - I)'].quantile(0.05)]


# In[ ]:





# In[ ]:




