#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:51:13 2019

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

region_select_ind = [1,2,6,8,10,13,15]
region_names_plt_sel = np.array(region_names_plt)
region_names_plt_sel = region_names_plt_sel[region_select_ind]
#select_ind = np.arange(0,len(region_names))
#Model output
filepath = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/MODELS/{model_name}/{model_type}/'.format(model_name=model_name,model_type=model_type)
#filenames = glob.glob(filepath+'STATS*.csv')#.format(region_name=region_namex))
filenames = glob.glob(filepath+'PCTILES_SIC_{no_days}day_change_{model_name}_ALL_ENS*.csv'.format(no_days=no_days,
                      model_name=model_name))
#Sort filenames too so region names and filenames are the same
filenames = sorted(filenames, key=str.lower)
#Same for obs
obs_fpath = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS'
fname_obs = glob.glob(obs_fpath+'/NSIDC_SIE_delta*_JJAS_NO_dt.txt')
fname_obs = sorted(fname_obs,key=str.lower)
#
pctile_list = [1,5,25,50,75,95,99]
for ifile in np.arange(0,len(filenames)):
#ifile = 12
#    #read in data
    print('region: ',region_names[ifile])
    print('model: ',filenames[ifile])
    print('obs: ',fname_obs[ifile])
#    #read in model putput
    i_file = filenames[ifile]
    i_data = pd.read_csv(i_file) #has shape of ensemble members x pctile (first column is throwaway)
    no_time = len(np.transpose(i_data))
    if ifile == 0:
        X_model = i_data.iloc[:,1:8]
    else:
        X_model = np.dstack((X_model,i_data.iloc[:,1:8]))   
#    i_data = i_data.iloc[:,26:no_time]           
#    #read in obs
    obs_read = pd.read_csv(fname_obs[ifile])
    obs_read = pd.DataFrame(obs_read.replace([np.inf,-np.inf],np.nan))
    obs_pctiles = np.percentile(obs_read,pctile_list)
    if ifile == 0:
        X_obs = obs_pctiles
    else:
        X_obs = np.vstack((X_obs,obs_pctiles))
#Loop through pctiles
fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/'
for ipct in np.arange(0,len(pctile_list)):
    ax,fig = make_boxplot(X_model[:,ipct,:],X_obs[:,ipct],region_names_plt)
    title_string = '{pctile}percentile, {seas}, {no_day}-day change'.format(pctile=pctile_list[ipct],
                    seas=seas_str,no_day=no_days)
    ax.set_title(title_string,fontsize=13)
    fname_save = fpath_save+'PCTILE_{pctile}_SIC_change_{noday}day_{model_name}_{model_type}_{season}'.format(pctile=pctile_list[ipct],
                                    noday=no_days,model_name=model_name,model_type=model_type,season=seas_str)
    plt.savefig(fname_save,format='png',dpi=600,bbox_inches='tight')
    #Same but for limited regions
    ax2,fig2 = make_boxplot(X_model[:,ipct,region_select_ind],X_obs[region_select_ind,ipct],region_names_plt_sel)
    title_string = '{pctile}percentile, {seas}, {no_day}-day change'.format(pctile=pctile_list[ipct],
                    seas=seas_str,no_day=no_days)
    ax2.set_title(title_string,fontsize=13)
    fname_save2 = fpath_save+'PCTILE_{pctile}_select_region_SIC_change_{noday}day_{model_name}_{model_type}_{season}'.format(pctile=pctile_list[ipct],
                                    noday=no_days,model_name=model_name,model_type=model_type,season=seas_str)
    plt.savefig(fname_save2,format='png',dpi=600,bbox_inches='tight')
#ax,fig = make_boxplot(i_data.iloc[:,1],obs_pctiles[1],region_names_plt[ifile])
#    obs_read.fillna(method='ffill')
#    obs_read.fillna(method='bfill')
#    obs_read = obs_read.iloc[:,0] 
#    no_ens = 25
#    no_bins = 25
#    
#    ax,fig = compare_histograms(i_data,obs_read,no_ens,no_bins)
#    title_string = '{region_name}, {season},  {no_day}-day change'.format(region_name=region_names[ifile],
#                    season=seas_str,no_day=no_days)
#    ax.set_title(title_string,fontsize=13)
#    fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/'
#    fname_save = fpath_save+'SIC_change_{noday}day_hist_compare_{model_name}_{model_type}_obs_{region_name}_{season}'.format(noday=no_days,
#                                        model_name=model_name,model_type=model_type,region_name=region_names[ifile],
#                                        season=seas_str)
#    plt.savefig(fname_save,format='png',dpi=600)
#    #plt.pause(2.5)
#    #ax.show()