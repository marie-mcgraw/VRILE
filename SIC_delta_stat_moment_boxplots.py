#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:15:13 2019

@author: mcmcgraw
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import xarray as xr
import os
import pandas as pd
import glob
import seaborn as sns

def make_boxplot(X_model,X_obs,region_names_plot):
    s1,s2 = X_model.shape
    
    if s1 == len(region_names_plot):
        X_model = np.transpose(X_model)
        
    medianprops = {'color': 'blue', 'linewidth': 2}
    boxprops = {'color': 'black','linestyle':'-', 'linewidth': 3}
    whiskerprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    capprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    flierprops = {'color': 'black', 'marker': 'o', 'markersize': 6, 'markeredgewidth': 2}
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    #fig.tight_layout()
    modelplot = ax.boxplot(X_model,medianprops=medianprops,boxprops=boxprops,
                            whiskerprops=whiskerprops,capprops=capprops,
                            flierprops=flierprops)
    obsplot = ax.plot(np.arange(1,len(region_names_plot)+1),X_obs,'rx',markersize=9,
                       markeredgewidth=2)
    ax.set_xticklabels(region_names_plot,rotation='vertical',fontsize=9)
    ax.legend(obsplot,['Observations'])
    ax.set_ylabel('Change in SIC (10^6 km^2)')
    
    return ax, fig

#Load model output and region names
model_name = 'ecmwfsipn'
model_type = 'reforecast'
no_days = 5 #number of days of sea ice change
seas_str = 'JJAS' #season
#select_ind = [0,6,7,14,15,16,17]
region_names = pd.read_csv('/home/disk/sipn/mcmcgraw/python/NSIDC_region_names.txt')
#Sort region names alphabetically
region_names = sorted(region_names.iloc[:,1],key=str.lower)

region_names_plt = ['BaffinBay','Barents','Barents-Kara','Beaufort','Bering','CanIslands','CentralArc',
                     'Chukchi','EGreenland','ESiberia','ESibBeaufChuk','HudsonBay','Kara','KaraLaptev',
                     'Laptev','panArctic','Okhotsk','StJohn']
#region_names_plt = region_names

#Model output
filepath = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/MODELS/{model_name}/{model_type}/'.format(model_name=model_name,model_type=model_type)
filenames = glob.glob(filepath+'STATS*JJAS.csv')#.format(region_name=region_namex))
#Sort filenames too so region names and filenames are the same
filenames = sorted(filenames, key=str.lower)
#Same for obs
obs_fpath = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS'
fname_obs = glob.glob(obs_fpath+'/NSIDC_SIE_delta*_JJAS_NO_dt.txt')
fname_obs = sorted(fname_obs,key=str.lower)
#Read in data for all regions
for ifile in np.arange(0,len(filenames)):
#read in data
#ifile = 0
    print('region: ',region_names[ifile])
    print('model: ',filenames[ifile])
    print('obs: ',fname_obs[ifile])
    #read in model output
    i_file = filenames[ifile]
    i_data = pd.read_csv(i_file)
    #For model output: MEAN, VARIANCE, SKEW, KURTOSIS
    if ifile == 0:
        MEAN_model = i_data['MEAN']
        VAR_model = i_data['VAR']
        SKEW_model = i_data['SKEW']
        KURT_model = i_data['KURTOSIS']
    else:
        MEAN_model = np.vstack((MEAN_model,i_data['MEAN']))
        VAR_model = np.vstack((VAR_model,i_data['VAR']))
        SKEW_model = np.vstack((SKEW_model,i_data['SKEW']))
        KURT_model = np.vstack((KURT_model,i_data['KURTOSIS']))
             
    #read in obs
    obs_read = pd.read_csv(fname_obs[ifile])
    obs_read = pd.DataFrame(obs_read.replace([np.inf,-np.inf],np.nan))
    obs_read.fillna(method='ffill')
    obs_read.fillna(method='bfill')
    obs_read = obs_read.iloc[:,0] 
    #Calculate moments for obs
    if ifile == 0:
        MEAN_obs = np.nanmean(obs_read)
        VAR_obs = np.nanvar(obs_read)
        SKEW_obs = stats.skew(obs_read,nan_policy='omit')
        KURT_obs = stats.kurtosis(obs_read,nan_policy='omit')
    else:
        MEAN_obs = np.vstack((MEAN_obs,np.nanmean(obs_read)))
        VAR_obs = np.vstack((VAR_obs,np.nanvar(obs_read)))
        SKEW_obs = np.vstack((SKEW_obs,stats.skew(obs_read,nan_policy='omit')))
        KURT_obs = np.vstack((KURT_obs,stats.kurtosis(obs_read,nan_policy='omit')))


fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/'
#Make plots
ax_M,fig_M = make_boxplot(MEAN_model[0:17,:],MEAN_obs[0:17],region_names_plt[0:17])
ax_M.set_title('Means')
fname_save = fpath_save+'MEAN_ALL_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_V,fig_V = make_boxplot(VAR_model[0:17,:],VAR_obs[0:17],region_names_plt[0:17])
ax_V.set_title('Variances')
fname_save = fpath_save+'VAR_ALL_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_S,fig_S = make_boxplot(SKEW_model[0:17,:],SKEW_obs[0:17],region_names_plt[0:17])
ax_S.set_title('Skewness')
fname_save = fpath_save+'SKEW_ALL_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_K,fig_K = make_boxplot(KURT_model[0:17,:],KURT_obs[0:17],region_names_plt[0:17])
ax_K.set_title('Kurtosis')
fname_save = fpath_save+'KURT_ALL_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
##Only areas of interest
select_regions_ind = [1,2,6,8,10,12,13,15]
select_rnames = np.array(region_names_plt)
ax_M,fig_M = make_boxplot(MEAN_model[select_regions_ind,:],MEAN_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_M.set_title('Means')
fname_save = fpath_save+'MEAN_select_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_V,fig_V = make_boxplot(VAR_model[select_regions_ind,:],VAR_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_V.set_title('Variances')
fname_save = fpath_save+'VAR_select_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_S,fig_S = make_boxplot(SKEW_model[select_regions_ind,:],SKEW_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_S.set_title('Skewness')
fname_save = fpath_save+'SKEW_select_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_K,fig_K = make_boxplot(KURT_model[select_regions_ind,:],KURT_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_K.set_title('Kurtosis')
fname_save = fpath_save+'KURT_select_regions_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#Areas of interest without panArctic
select_regions_ind = [1,2,6,8,10,12,13]
ax_M,fig_M = make_boxplot(MEAN_model[select_regions_ind,:],MEAN_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_M.set_title('Means')
fname_save = fpath_save+'MEAN_select_regions_no_pan_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_V,fig_V = make_boxplot(VAR_model[select_regions_ind,:],VAR_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_V.set_title('Variances')
fname_save = fpath_save+'VAR_select_regions_no_pan_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_S,fig_S = make_boxplot(SKEW_model[select_regions_ind,:],SKEW_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_S.set_title('Skewness')
fname_save = fpath_save+'SKEWNESS_select_regions_no_pan_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                        model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")
#
ax_K,fig_K = make_boxplot(KURT_model[select_regions_ind,:],KURT_obs[select_regions_ind],select_rnames[select_regions_ind])
ax_K.set_title('Kurtosis')
fname_save = fpath_save+'KURTOSIS_select_regions_no_pan_SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{season}'.format(noday=no_days,
                                       model_name=model_name,model_type=model_type,season=seas_str)
plt.savefig(fname_save,format='png',dpi=600,bbox_inches = "tight")