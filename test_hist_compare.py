#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import xarray as xr
import os
import pandas as pd
import glob

model_name = 'ecmwfsipn'
model_type = 'reforecast'
filepath = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/MODELS/{model_name}/{model_type}/'.format(model_name=model_name,model_type=model_type)
filenames = glob.glob(filepath+'RAW*_pan*.csv')
print(filenames)


# In[18]:


X_model = pd.read_csv(filenames[0])
X_model = X_model.iloc[:,26:]


# In[19]:


fpath_obs = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS'
fname_obs = glob.glob(fpath_obs+'/NSIDC_SIE_delta*pan*JJAS_NO_dt.txt')
print(fname_obs)
X_obs = pd.read_csv(fname_obs[0])


# In[49]:


bins = np.linspace(-0.6,0.6,50)
#Remove nans from X_obs
X_obs = pd.DataFrame(X_obs.replace([np.inf,-np.inf],np.nan))
X_obs.fillna(method='ffill')
X_obs.fillna(method='bfill')
print(X_obs.shape)
H_obs = stats.gaussian_kde(X_obs.iloc[1:,0])
H_obs_plot = H_obs.evaluate(bins)

#Model
no_ens = 25
for iens in np.arange(0,no_ens):
    X_idat = X_model.iloc[iens,:]
    H_mod = stats.gaussian_kde(X_idat)
    H_mod_plot = H_mod.evaluate(bins)
    if iens == 0:
        H_mod_ALL = H_mod
        H_mod_plot_ALL = H_mod_plot
    else:
        H_mod_ALL = np.vstack((H_mod_ALL,H_mod))
        H_mod_plot_ALL = np.vstack((H_mod_plot_ALL,H_mod_plot))
        
print(H_mod_plot_ALL.shape)  


# In[53]:


print(H_obs_plot)
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(bins,H_obs_plot,'b',linewidth=2)
ax1.plot(bins,np.nanmean(H_mod_plot_ALL,axis=0),'k')
ax1.plot(bins,np.amax(H_mod_plot_ALL,axis=0),'xkcd:salmon')
ax1.plot(bins,np.amin(H_mod_plot_ALL,axis=0),'xkcd:salmon')


# In[ ]:




