#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:37:41 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from datetime import datetime, timedelta
import statsmodels.api as sm

def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvar]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

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
    
    
model_name = 'ecmwfsipn'
model_type = 'reforecast'
no_ens = 25 #25 ensemble members
no_day_change = 5 #looking at 5 day changes
day_change = no_day_change
max_lead = 30

delta_lead_days = np.arange(no_day_change,max_lead+1,1)
delta_first_days = np.arange(1,max_lead+2-no_day_change,1)
no_forecast_periods = len(delta_lead_days)

filepath_load = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
filename_full = filepath_load+'MOVING_{model_name}_{model_type}_d_SIC_{d_days}day_change_lead_time_{lead_days}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                               model_type=model_type,d_days=no_day_change,
                               lead_days=no_day_change*no_forecast_periods)

file_full = pd.read_csv(filename_full)
regions_list = file_full['region'].unique().tolist()
valid_dates_list = file_full['V (valid date)'].unique().tolist()
all_dates = pd.to_datetime(file_full['V (valid date)'])
#all_dates_months = all_dates.dt.month
#all_dates_years = all_dates.dt.year
years_unique = all_dates.dt.year.unique().tolist()
no_yrs = len(years_unique)
mon_sel = [6,7,8,9] #pick the season
seas_str = 'JJAS'
#Group data by regions
SIE_regions = file_full.groupby(['region'])
d_SIE_trends_rolling = pd.DataFrame(columns=["region",
                                             "ensemble",
                                             "year",
                                             "season",
                                             "5th pctile",
                                             "trend (10^6 km^2/yr)",
                                             "trend int (10^6 km^2)",
                                             ])
#Loop through regions
#ireg = 0
for ireg in np.arange(0,len(regions_list)):    
    region_name = regions_list[ireg]
    sie_ireg = SIE_regions.get_group(region_name)
    df_inside_reg = pd.DataFrame(columns=["region",
                                  "ensemble",
                                  "year",
                                  "season",
                                  "5th pctile",
                                  "trend (10^6 km^2/yr)",
                                  "trend int (10^6 km^2)"])
    av_period = 2
    no_yrs_moving = len(years_unique[av_period:-av_period])
    no_ens = 25
    #Now group by ensemble
    SIE_ens = sie_ireg.groupby(['ensemble'])
    #iens = 1
    for iens in np.arange(1,no_ens+1):
        ens_save_ind = (iens-1)*no_yrs_moving + np.arange(0,no_yrs_moving)
        sie_iens = SIE_ens.get_group(iens)
        select_valid = sie_iens['V (valid date)']
        dates_select = pd.to_datetime(select_valid)
        df_inside_ens = pd.DataFrame(columns=["region",
                                      "ensemble",
                                      "year",
                                      "season",
                                      "5th pctile",
                                      "trend (10^6 km^2/yr)",
                                      "trend int (10^6 km^2)"])
        #now we loop through years
        p5 = np.array([])
        trend_slope = np.array([])
        trend_int = np.array([])
        cent_year = list()
        for yr_ind in np.arange(av_period,len(years_unique)-av_period):
        #yr_ind = 2
            year_sel = np.arange(years_unique[yr_ind]-av_period,years_unique[yr_ind]+av_period+1)
            #
            cent_year.append(year_sel[av_period])
            choose_dates = (dates_select.dt.year.isin(year_sel)) & (dates_select.dt.month.isin(mon_sel))
            SIE_select = sie_iens['d_SIC (V - I)'].where(choose_dates == True)
            p5 = np.append(p5,np.nanpercentile(SIE_select,0.05))
            no_nans = np.isfinite(SIE_select)
            pf = np.polyfit(np.arange(0,len(SIE_select[no_nans])),SIE_select[no_nans],1)    
            trend_slope = np.append(trend_slope,pf[0])
            trend_int = np.append(trend_int,pf[1])
            
        df_inside_ens['5th pctile'] = p5
        df_inside_ens['ensemble'] = np.tile(iens,len(p5))
        df_inside_ens['season'] = np.tile(seas_str,len(p5))
        df_inside_ens['region'] = np.tile(region_name,len(p5))
        df_inside_ens['year'] = cent_year
        df_inside_ens['trend (10^6 km^2/yr)'] = trend_slope
        df_inside_ens['trend int (10^6 km^2)'] = trend_int
        
        if iens == 1:
            df_inside_reg = df_inside_ens
        else:
            df_inside_reg = df_inside_reg.append(df_inside_ens)
            
    if ireg == 0:
        d_SIE_trends_rolling = df_inside_reg
    else:
        d_SIE_trends_rolling = d_SIE_trends_rolling.append(df_inside_reg)
        
fpath_save = '/home/disk/sipn/mcmcgraw/data/VRILE/'
fname_save = fpath_save+'trends_5th_ptiles_ALL_regions_rolling_{noyrs}yrs.csv'.format(noyrs=2*av_period+1)
d_SIE_trends_rolling.to_csv(fname_save)

trends_region = d_SIE_trends_rolling.groupby(['region'])
for ireg_plot in np.arange(0,len(regions_list)):
    region_plot = ireg_plot
    trends_group = trends_region.get_group(regions_list[region_plot])
    trends_v_time = trends_group.groupby(['year'])
    trends_v_ens = trends_group.groupby(['ensemble'])
    cent_yrs_plot = d_SIE_trends_rolling['year'].unique().tolist()
        
    colors_plot = ['#800000','#9A6324','#808000','#469990','#000075',
     '#000000','#e6194B','#f58231','#ffe119','#bfef45',
     '#3cb44b','#42d4f4','#4363d8','#911eb4','#f032e6',
     '#a9a9a9','#fabebe','#ffd8b1','#fffac8','#aaffc3',
     '#e6beff','#a87900','#236B8E','#871F78','#38B0DE']
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])
    
    fig3 = plt.figure()
    ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])
    #p5_all = np.array([])
    for iens_plt in np.arange(0,no_ens):
        p5_plt = trends_v_ens['5th pctile'].get_group(iens_plt+1)
        trnd_plt = trends_v_ens['trend (10^6 km^2/yr)'].get_group(iens_plt+1)
        if iens_plt == 0:
            p5_all = p5_plt
            trnd_all = trnd_plt
        else:
            p5_all = np.vstack((p5_all,p5_plt))
            trnd_all = np.vstack((trnd_all,trnd_plt))
        ax1.plot(p5_plt,color=colors_plot[iens_plt])
        ax3.plot(trnd_plt,color=colors_plot[iens_plt])
        
    ax1.plot(np.nanmean(p5_all,0),color='k',linewidth=3)
    ax1.set_ylabel('change in SIE (10^6 km^2)',fontsize=11)
    ax1.set_xticks(np.arange(0,len(p5_plt),2))
    ax1.set_xticklabels(cent_yrs_plot[1::2])
    ax1.set_title('5th pctile, {noyr}-year av, {region}'.format(noyr=2*av_period+1,
                  region=regions_list[region_plot]))
    fname_1 = '/home/disk/sipn/mcmcgraw/figures/VRILE/trends_5thpctiles/{model_name}_{model_type}_all_ens_5th_pctile_{region}_{noyr}yr_rolling.png'.format(model_name=model_name,
                                                                        model_type=model_type,region=regions_list[region_plot],
                                                                        noyr=av_period*2+1)
    fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
    ax3.plot(np.nanmean(trnd_all,0),color='k',linewidth=3)
    ax3.set_ylabel('change in SIE (10^6 km^2)/{noyr} yrs'.format(noyr=2*av_period+1),fontsize=11)
    ax3.set_xticks(np.arange(0,len(p5_plt),2))
    ax3.set_xticklabels(cent_yrs_plot[1::2])
    ax3.set_title('trend in mean, {noyr}-year av, {region}'.format(noyr=2*av_period+1,
                  region=regions_list[region_plot]))
    fname_3 = '/home/disk/sipn/mcmcgraw/figures/VRILE/trends_5thpctiles/{model_name}_{model_type}_envelope_5th_pctile_{region}_{noyr}yr_rolling.png'.format(model_name=model_name,
                                                                        model_type=model_type,region=regions_list[region_plot],
                                                                        noyr=av_period*2+1)
    fig3.savefig(fname_3,format='png',dpi=600,bbox_inches='tight')
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
    ax2.fill_between(np.arange(0,len(np.nanmean(p5_all,0))),np.nanmin(p5_all,0),np.nanmax(p5_all,0),color='xkcd:coral')
    ax2.plot(np.nanmean(p5_all,0),color='k',linewidth=3)
    ax2.set_ylabel('change in SIE (10^6 km^2)',fontsize=11)
    ax2.set_xticks(np.arange(0,len(p5_plt),2))
    ax2.set_xticklabels(cent_yrs_plot[1::2])
    ax2.set_title('5th pctile, {noyr}-year av, {region}'.format(noyr=2*av_period+1,
                  region=regions_list[region_plot]))
    fname_2 = '/home/disk/sipn/mcmcgraw/figures/VRILE/trends_5thpctiles/{model_name}_{model_type}_all_ens_trend_{region}_{noyr}yr_rolling.png'.format(model_name=model_name,
                                                                        model_type=model_type,region=regions_list[region_plot],
                                                                        noyr=av_period*2+1)
    fig2.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')
    #
    fig4 = plt.figure()
    ax4 = fig4.add_axes([0.1,0.1,0.8,0.8])
    ax4.fill_between(np.arange(0,len(np.nanmean(trnd_all,0))),np.nanmin(trnd_all,0),np.nanmax(trnd_all,0),color='xkcd:coral')
    ax4.plot(np.nanmean(trnd_all,0),color='k',linewidth=3)
    ax4.set_ylabel('change in SIE (10^6 km^2)/{noyr} yrs'.format(noyr=2*av_period+1),fontsize=11)
    ax4.set_xticks(np.arange(0,len(p5_plt),2))
    ax4.set_xticklabels(cent_yrs_plot[1::2])
    ax4.set_title('trend in mean, {noyr}-year av, {region}'.format(noyr=2*av_period+1,
                  region=regions_list[region_plot]))
    fname_4 = '/home/disk/sipn/mcmcgraw/figures/VRILE/trends_5thpctiles/{model_name}_{model_type}_envelope_trend_{region}_{noyr}yr_rolling.png'.format(model_name=model_name,
                                                                        model_type=model_type,region=regions_list[region_plot],
                                                                        noyr=av_period*2+1)
    fig4.savefig(fname_4,format='png',dpi=600,bbox_inches='tight')