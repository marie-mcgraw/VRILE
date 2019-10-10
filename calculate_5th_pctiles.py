#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script calculates and plots basic information about the 5th percentile events of 5-day 
# sea ice extent change (SIE).  The 5th percentile of 5-day change in SIE is our threshold for 
# defining very rapid ice loss events (VRILEs)
"""
Created on Tue Sep  3 15:00:11 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from datetime import datetime, date
import scipy.stats as stats

##Clear workspace before running 
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

#Load model sea ice extent (SIE) data    
filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
model_name = 'ecmwfsipn'
model_type = 'reforecast'
day_change = 5
max_lead = 30

filename = 'MOVING_{model_name}_{model_type}_d_SIC_{day_change}day_change_lead_time_1{max_lead}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                                                                                                                                 model_type=model_type,
                                                                                                                                 day_change=day_change,
                                                                                                                                 max_lead=max_lead)
#filename2 = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/ecmwfsipn_reforecast_d_SIC_5day_change_lead_time_30days_ALL_REGIONS_ALL_ENS.csv'
ds_SIC_all = pd.read_csv(filepath+filename)
pctile_select = 5 #look at 5th percentile
regions = ds_SIC_all['region']
#region_names = ['panArctic','East Greenland Sea','Barents Sea','Central Arctic',
 #               'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
region_names = ds_SIC_all['region'].unique().tolist()

#Create column names for output, which will be stored in Pandas dataframes
column_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Region']
mon_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#initialize dataframe for 5th percentile from model data
SIC_p5 = pd.DataFrame(columns=column_names)
#Now, load data from NSIDC (observations)
#region_names_obs = ['panArctic','EastGreenlandSea','BarentsSea','CentralArctic',
#                'Kara-Laptev','East-Siberian-Beaufort-Chukchi']
region_names_obs = [iname.replace(' ','') for iname in region_names]
filepath_obs = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS/'  #observed sea ice extent data
fname_time_obs = 'NSIDC_SIE_delta_TIME_5day_change_ALL_NO_dt.csv'
time_obs = pd.read_csv(filepath_obs+fname_time_obs)
obs_year = time_obs['year']
obs_month = time_obs['month']
#initialize more dataframes
SIC_obs_p5 = pd.DataFrame(columns=column_names) #5th percentile, observed data
SIC_model_less_obs_p5 = pd.DataFrame(columns=column_names) #pct of time that modeled SIE is below observed 5th percentile
SIC_model_p5 = pd.DataFrame(columns=column_names) #pct of time that modeled SIE is below observed 5th percentile (should be about 5 if modeled SIE changes are gaussian)
SIC_obs_median = pd.DataFrame(columns=column_names)
SIC_obs_mean = pd.DataFrame(columns=column_names)
SIC_model_median = pd.DataFrame(columns=column_names)
SIC_model_mean = pd.DataFrame(columns=column_names)
SIC_obs_p95 = pd.DataFrame(columns=column_names)
SIC_model_p95 = pd.DataFrame(columns=column_names)
SIC_obs_var = pd.DataFrame(columns=column_names)
SIC_model_var = pd.DataFrame(columns=column_names)
#Loop through each region separately
for ireg in np.arange(0,len(region_names)):
    #ireg = 0
    iname = region_names[ireg]
    #Select only d_SI values for that region
    region_sel = regions.index.where(regions==iname)
    ds_SIC_ireg = ds_SIC_all.iloc[~np.isnan(region_sel),:]
    #Convert initialization and valid dates to datetime objects
    ds_SIC_ireg['I (init date)'] = pd.to_datetime(ds_SIC_ireg['I (init date)'])
    init_date2 = ds_SIC_ireg['I (init date)']
    ds_SIC_ireg['V (valid date)'] = pd.to_datetime(ds_SIC_ireg['V (valid date)'])
    valid_date = ds_SIC_ireg['V (valid date)']
    dSI = ds_SIC_ireg['d_SIC (V - I)']
    #Select observed data for specified region
    fname_obs = 'NSIDC_SIE_delta_{day_change}day_change_{region}_ALL_NO_dt.txt'.format(day_change=day_change,
                     region=region_names_obs[ireg])
    SIC_obs = pd.read_csv(filepath_obs+fname_obs)
    ##Filter obs--do things change?
    filt = True
    filt_app = 'NO_FILT' #used for filenames
    if filt == True:
        SIC_obs = SIC_obs.rolling(5).mean()
        filt_app = 'FILT'
    #Now, separate by month
    for imon in np.arange(1,13):
        mon_sel_ind = valid_date.index.where(valid_date.dt.month == imon) #select correct month for model
        d_SIC_mon_sel = dSI.where(~np.isnan(mon_sel_ind)) #model data corresponding to that month--all ensembles
        d_SIC_mon_sel = d_SIC_mon_sel.dropna()#drop NaNs so we actually know sample size
        mean = np.nanmean(d_SIC_mon_sel)
        med = np.nanmedian(d_SIC_mon_sel)
        imon_name = mon_names[imon-1] #name of month
        d_SIC_mon_sel = d_SIC_mon_sel - np.nanmean(d_SIC_mon_sel) #remove mean
        p5 = np.nanpercentile(d_SIC_mon_sel,pctile_select) #calculate 5th pctile
        p95 = np.nanpercentile(d_SIC_mon_sel,100-pctile_select)
        #write to dataframe
        SIC_p5.loc[ireg,imon_name] = p5
        SIC_p5.loc[ireg,'Region'] = iname
        SIC_model_mean.loc[ireg,imon_name] = mean
        SIC_model_median.loc[ireg,imon_name] = med
        SIC_model_mean.loc[ireg,'Region'] = iname
        SIC_model_median.loc[ireg,'Region'] = iname
        SIC_model_p95.loc[ireg,imon_name] = p95
        SIC_model_p95.loc[ireg,'Region'] = iname
        SIC_model_var.loc[ireg,imon_name] = np.nanvar(d_SIC_mon_sel)
        SIC_model_var.loc[ireg,'Region'] = iname
        #Obs--select obs for each month
        obs_sel_ind = obs_month.index.where(obs_month == imon)
        obs_sel_ind = obs_sel_ind[0:-5]
        d_SIC_obs_sel = SIC_obs[~np.isnan(obs_sel_ind)]
        mean_obs = np.nanmean(d_SIC_obs_sel)
        med_obs = np.nanmedian(d_SIC_obs_sel)
        d_SIC_obs_sel = d_SIC_obs_sel - np.nanmean(d_SIC_obs_sel)
        p5_obs = np.nanpercentile(d_SIC_obs_sel,pctile_select)
        p95_obs = np.nanpercentile(d_SIC_obs_sel,100-pctile_select)
        SIC_obs_p5.loc[ireg,imon_name] = p5_obs
        SIC_obs_p5.loc[ireg,'Region'] = iname
        SIC_obs_mean.loc[ireg,imon_name] = mean_obs
        SIC_obs_median.loc[ireg,imon_name] = med_obs
        SIC_obs_mean.loc[ireg,'Region'] = iname
        SIC_obs_median.loc[ireg,'Region'] = iname
        SIC_obs_p95.loc[ireg,imon_name] = p95_obs
        SIC_obs_p95.loc[ireg,'Region'] = iname
        SIC_obs_var.loc[ireg,imon_name] = np.nanvar(d_SIC_obs_sel)
        SIC_obs_var.loc[ireg,'Region'] = iname
        #Number of times that the model is less than the obs 5th pctile
        model_x = d_SIC_mon_sel.where(d_SIC_mon_sel <= p5_obs)
        SIC_model_less_obs_p5.loc[ireg,imon_name] = 100*(model_x.count()/d_SIC_mon_sel.count())
        SIC_model_less_obs_p5.loc[ireg,'Region'] = iname
        model_5 = d_SIC_mon_sel.where(d_SIC_mon_sel <= p5)
        SIC_model_p5.loc[ireg,imon_name] = 100*(model_5.count()/d_SIC_mon_sel.count())
        SIC_model_p5.loc[ireg,'Region'] = iname

fname_save = '{model_name}_{model_type}_5th_pctile_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS.csv'.format(model_name=model_name,
              model_type=model_type,day_change=day_change,max_lead=max_lead)
SIC_p5.to_csv('/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'+fname_save)    

SIC_model_mean.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                     "{model_name}_{model_type}_mean_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS.csv".format(model_name=model_name,
              model_type=model_type,day_change=day_change,max_lead=max_lead))    
#
SIC_model_median.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                      "{model_name}_{model_type}_median_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS.csv".format(model_name=model_name,
              model_type=model_type,day_change=day_change,max_lead=max_lead))
#
SIC_model_p95.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                      "{model_name}_{model_type}_95th_pctile_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS.csv".format(model_name=model_name,
              model_type=model_type,day_change=day_change,max_lead=max_lead)) 

SIC_model_var.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                      "{model_name}_{model_type}_variance_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS.csv".format(model_name=model_name,
              model_type=model_type,day_change=day_change,max_lead=max_lead))   
#for xreg in np.arange(0,len(region_names_obs)):
#xreg = 0
#fname_obs = 'NSIDC_SIE_delta_{day_change}day_change_{region}_ALL_NO_dt.txt'.format(day_change=day_change,
#                     region=region_names_obs[xreg])
#SIC_obs = pd.read_csv(filepath_obs+fname_obs)
#    for xmon in np.arange(1,13):  
#xmon = 1      
#obs_sel_ind = obs_month.index.where(obs_month == xmon)
#obs_sel_ind = obs_sel_ind[0:-5]
#d_SIC_obs_sel = SIC_obs[~np.isnan(obs_sel_ind)]
#d_SIC_obs_sel = d_SIC_obs_sel - np.nanmean(d_SIC_obs_sel)
#p5_obs = np.nanpercentile(d_SIC_obs_sel,pctile_select)
#xmon_name = mon_names[xmon-1]
#SIC_obs_p5.loc[xreg,xmon_name] = p5_obs
#SIC_obs_p5.loc[xreg,'Region'] = region_names[xreg]



fname_obs_save = 'NSIDC_5th_pctile_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS_{filt_app}.csv'.format(day_change=day_change,
                                   max_lead=max_lead,filt_app=filt_app)
SIC_obs_p5.to_csv('/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'+fname_obs_save)

SIC_obs_mean.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                    "NSIDC_mean_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS_{filt_app}.csv".format(day_change=day_change,
                                   max_lead=max_lead,filt_app=filt_app))
SIC_obs_median.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                    "NSIDC_median_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS_{filt_app}.csv".format(day_change=day_change,
                                   max_lead=max_lead,filt_app=filt_app))
SIC_obs_p95.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                    "NSIDC_95th_pctile_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS_{filt_app}.csv".format(day_change=day_change,
                                   max_lead=max_lead,filt_app=filt_app))
#
SIC_obs_var.to_csv("/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/"\
                    "NSIDC_variance_{day_change}day_change_lead_time_{max_lead}_days_ALL_ENS_{filt_app}.csv".format(day_change=day_change,
                                   max_lead=max_lead,filt_app=filt_app))
#colors = ['xkcd:bright blue','xkcd:salmon','xkcd:goldenrod','xkcd:red orange',
#          'xkcd:barney purple','xkcd:dark lime green']
colors = ['#7e1e9c','#15b01a','#0343df','#ff81c0','#653700','#e50000',
          '#030aa7','#06470c','#000000','#dbb40c','#9e3623','#5a86ad',
          '#75bbfd','#929591','#f97306','#c20078','#04d8b2','#840000']
          
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])

fig3 = plt.figure()
ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])

d_obs_minus_model = SIC_obs_p5.iloc[:,0:12] - SIC_p5.iloc[:,0:12]

leg_handles = np.array([])
for iplot in np.arange(0,len(region_names)):
    h0 = ax1.plot(SIC_p5.iloc[iplot,0:12],color=colors[iplot],linewidth=1.3,marker='o',markersize=4)
    leg_handles = np.append(leg_handles,h0)
    ax1.plot(SIC_obs_p5.iloc[iplot,0:12],color=colors[iplot],linestyle=':',linewidth=1.3,marker='s',markersize=4)
    
    ax2.plot(d_obs_minus_model.iloc[iplot,:],color=colors[iplot],linewidth=1.3,marker='o',markersize=4)
    #
    ax3.plot(SIC_model_less_obs_p5.iloc[iplot,0:12],color=colors[iplot],linestyle='-.',linewidth=1.3,marker='s',markersize=4)
    #ax3.plot(SIC_model_p5.iloc[iplot,0:12],color=colors[iplot],linewidth=1.3,marker='o',markersize=4)
    

ax1.axvspan(4.85,8.15,alpha=0.4,color='gray')    
ax1.legend(leg_handles,region_names,ncol=2,bbox_to_anchor=(1, 1.025))
ax1.set_ylabel('Change in SIE (10^6 km^2)',fontsize=11)
ax1.set_title('Magnitude of 5th Percentile, Lead Time up to {max_lead} Days'.format(max_lead=max_lead))
fname_1 = '/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_AND_obs_pctile5_{day_change}day_change_{max_lead}day_lead_{filt_app}.png'.format(model_name=model_name,
                                                  model_type=model_type,day_change=day_change,max_lead=max_lead,
                                                  filt_app=filt_app)
fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')

ax2.axvspan(4.85,8.15, alpha=0.4, color='gray')
ax2.plot(np.arange(-1,13),np.zeros(14,),color='k',linewidth=2)
ax2.set_ylim([-0.06,0.06])
ax2.legend(region_names,ncol=2,bbox_to_anchor=(1.025,0.52165))
ax2.set_ylabel('Change in SIE (10^6 km^2)')
ax2.set_title('5th Percentile, Obs - Model, lead time {max_lead} days'.format(max_lead=max_lead))
ax2.set_xlim([-0.5,11.5])
fname_2 = '/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_model_minus_obs_pctile5_{day_change}day_change_{max_lead}day_lead_{filt_app}.png'.format(model_name=model_name,
                                                  model_type=model_type,day_change=day_change,
                                                  max_lead=max_lead,filt_app=filt_app)
fig2.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')

#
ax3.plot(np.arange(-1,13),5*np.ones(14),color='k',linewidth=1.5)
ax3.axvspan(4.85,8.15,alpha=0.4,color='gray')    
ax3.set_xlim(-0.5,11.5)
ax3.legend(leg_handles,region_names,ncol=2,bbox_to_anchor=(1, 1.025))
ax3.set_ylabel('%',fontsize=11)
ax3.set_title('Frequency that modeled SIE is below observed 5th percentile',fontsize=12)
fname_3 = '/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_freq_model_below_obs_5pct_{day_change}day_change_{max_lead}day_lead_{filt_app}.png'.format(model_name=model_name,
    model_type=model_type,day_change=day_change,max_lead=max_lead,filt_app=filt_app)
    
fig3.savefig(fname_3,format='png',dpi=600,bbox_inches='tight')

region_names_sub = [0,6,7,13,14,15,16]

fig4 = plt.figure()
ax4 = fig4.add_axes([0.1,0.1,0.8,0.8])

fig5 = plt.figure()
ax5 = fig5.add_axes([0.1,0.1,0.8,0.8])

for iplotx in region_names_sub:
    h4 = ax4.plot(SIC_model_less_obs_p5.iloc[iplotx,0:12],color=colors[iplotx],linestyle='-.',
                  linewidth=1.3,marker='s',markersize=4)
    h5 = ax5.plot(SIC_p5.iloc[iplotx,0:12],color=colors[iplotx],linewidth=1.3,
                  marker='o',markersize=4)
    ax5.plot(SIC_obs_p5.iloc[iplotx,0:12],color=colors[iplotx],linestyle=':',
             linewidth=1.3,marker='s',markersize=4)
    
ax4.plot(np.arange(-1,13),5*np.ones(14),color='k',linewidth=1.5)
ax4.axvspan(4.85,8.15,alpha=0.4,color='gray')    
ax4.set_xlim(-0.5,11.5)
ax4.legend(leg_handles[region_names_sub],[region_names[regind] for regind in region_names_sub],bbox_to_anchor=(1,1))
ax4.set_ylabel('%',fontsize=11)
ax4.set_title('Frequency that modeled SIE is below observed 5th percentile',fontsize=12)
fname_4 = "/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_"\
"_freq_model_below_obs_5pct_SELECT_{day_change}day_change_{max_lead}day_lead_"\
"_{filt_app}.png".format(model_name=model_name,model_type=model_type,
  day_change=day_change,max_lead=max_lead,filt_app=filt_app)
fig4.savefig(fname_4,format='png',dpi=600,bbox_inches='tight')
#ax4.set_ylim([-0])

ax4.set_ylim(-0.5,15.5)
fname_4b = "/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_"\
"_freq_model_below_obs_5pct_SELECT_ZOOM_{day_change}day_change_{max_lead}day_lead_"\
"_{filt_app}.png".format(model_name=model_name,model_type=model_type,
  day_change=day_change,max_lead=max_lead,filt_app=filt_app)
fig4.savefig(fname_4b,format='png',dpi=600,bbox_inches='tight')

#
ax5.axvspan(4.85,8.15,alpha=0.4,color='gray')    
ax5.legend(leg_handles[region_names_sub],[region_names[regind2] for regind2 in region_names_sub],
           bbox_to_anchor=(1, 1))
ax5.set_ylabel('Change in SIE (10^6 km^2)',fontsize=11)
ax5.set_title('Magnitude of 5th Percentile, Lead Time up to {max_lead} Days'.format(max_lead=max_lead))
fname_5 = '/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_AND_obs_SELECT_pctile5_{day_change}day_change_{max_lead}day_lead_{filt_app}.png'.format(model_name=model_name,
                                                  model_type=model_type,day_change=day_change,max_lead=max_lead,
                                                  filt_app=filt_app)
fig5.savefig(fname_5,format='png',dpi=600,bbox_inches='tight')