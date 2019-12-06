#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import pandas as pd

# Create plotting function

# Import our SIE data set, which we saved as a .csv file.  Use the head command to show all of our data. 

# In[2]:


model_name = 'ecmwfsipn'
model_type = 'reforecast'
no_day_change = 5
filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/'
filename = filepath+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                       model_type=model_type,d_days=no_day_change)
SIE_data = pd.read_csv(filename)
SIE_data = SIE_data.dropna()
SIE_data.head()

# Do same for obs
obs_name = 'NSIDC_0079'
obs_type = 'sipn_nc_yearly_agg'
obs_filename = filepath+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=obs_name,
                       model_type=obs_type,d_days=no_day_change)
SIE_obs = pd.read_csv(obs_filename)
SIE_obs = SIE_obs.dropna()
SIE_obs.head()
# Make histograms grouped by region, lead day, and valid date. Start with regions.  Use groupby to group dataframe by regions, and unique() to get the list of regions

# In[3]:


region_names = SIE_data['region'].dropna().unique().tolist()
lead_days_select = [7,14,21,28,35,42,49,56]
no_lead = len(lead_days_select)
no_tiles = no_lead*len(region_names)
#ks_failures = pd.DataFrame(columns={"lead day":np.tile(lead_days_select,len(region_names)),
#                                     "region":"",
#                                     "SIE no. failures":"",
#                                     "d_SIE no. failures":""})
ks_failures = pd.DataFrame(columns=['lead day','region','SIE no. failures','d_SIE no. failures'])
ks_failures['lead day'] = np.tile(lead_days_select,len(region_names))
#%%
##Remove 1989-1992 and 2017-2018
obs_yr = pd.to_datetime(SIE_obs['V (valid date)']).dt.year
yrmin = 1993
yrmax = 2017
yr_vec = np.arange(yrmin,yrmax)
f_yr_obs = np.isin(obs_yr,yr_vec) 
SIE_obs_trim = SIE_obs.loc[f_yr_obs]

#%%
for ireg in np.arange(0,len(region_names)):
#ireg = 7
    region_ind_test = ireg
    region_name_test = region_names[region_ind_test]
    SIE_region_groupby = SIE_data.groupby(['region'])
    SIE_by_region = SIE_region_groupby.get_group(region_name_test)
    SIEobs_region_groupby = SIE_obs_trim.groupby(['region'])
    SIEobs_by_region = SIEobs_region_groupby.get_group(region_name_test)
    
    # Now select for month of valid date (JJAS for now)
    
    # In[29]:
    
    
    valid_dates = pd.to_datetime(SIE_by_region['V (valid date)'])
    valid_days_months = pd.to_datetime(SIE_by_region['V (valid date)']).dt.month
    obs_valid_months = SIEobs_by_region['V_mon (valid date month)']
    #SIE_valid_groupby = SIE_by_region.groupby(['V (valid date)'])
    month_sel = [6,7,8,9]
    month_sel_str = 'JJAS'
    #choose_by_month = SIE_data['V (valid date)'].isin((valid_days_months,month_sel))
    #print(valid_days_months)
    f_month = np.isin(valid_days_months.values,month_sel)
    f_month_obs = np.isin(obs_valid_months,month_sel)
    SIE_month_trim = SIE_by_region.loc[f_month]
    SIE_obs_trim = SIEobs_by_region.loc[f_month_obs]
    
    # Now, filter by lead date.  We'll start with lead times of 1-4 weeks
    
    # In[30]:
    
    
    SIE_lead_groups = SIE_month_trim.groupby(['lead time (days)'])
    #ilead = 0
    for ilead in np.arange(0,len(lead_days_select)):
        lead_days_ind = ilead
        SIE_lead_sel = SIE_lead_groups.get_group(lead_days_select[ilead])
        is_detrend = False
        if is_detrend == True:
            SIE_dt = signal.detrend(SIE_lead_sel['SIE'])
            SIEobs_dt = signal.detrend(SIE_obs_trim['SIE'])
            SIE_lead_sel['SIE'] = SIE_dt
            SIE_obs_trim['SIE'] = SIEobs_dt
        #plt.plot(SIE_lead_sel['SIE'])
        #plt.plot(SIE_lead_sel['SIE'],color='r')
        
        
        # Remove seasonality.  We'll just do it by month--remove mean of all June days from each June day, etc. Add a new column to the existing dataframe with seasonal anomalies of SIE, and one with seasonal anomalies of the change in SIE
        
        # In[31]:
        
        
        SIE_seas_rem = np.array([])
        d_SIE_seas_rem = np.array([])
        valid_months_lead_sel = pd.to_datetime(SIE_lead_sel['V (valid date)']).dt.month
        valid_dates_lead_sel = pd.to_datetime(SIE_lead_sel['V (valid date)'])
        
        for i_SIE in np.arange(0,len(SIE_lead_sel)):
        #i_SIE = 100
        #print(valid_months_lead_sel.iloc[0])
            mon_i = valid_months_lead_sel.iloc[i_SIE]
            all_mon_i = np.where(valid_months_lead_sel == mon_i)
            i_ice = SIE_lead_sel['SIE'].iloc[i_SIE] - SIE_lead_sel['SIE'].iloc[all_mon_i].mean()
            i_d_ice = SIE_lead_sel['d_SIE (V - I)'].iloc[i_SIE] - SIE_lead_sel['d_SIE (V - I)'].iloc[all_mon_i].mean()
            SIE_seas_rem = np.append(SIE_seas_rem,i_ice)
            d_SIE_seas_rem = np.append(d_SIE_seas_rem,i_d_ice)
        print(SIE_seas_rem)
        print(d_SIE_seas_rem)
        SIE_lead_sel['SIE (seas rem)'] = SIE_seas_rem
        SIE_lead_sel['d_SIE (seas rem)'] = d_SIE_seas_rem
        SIE_lead_sel.head()
        
        # Now, we'll remove the seasonal cycle impacts, but for obs
        SIE_obs_seas_rem = np.array([])
        d_SIE_obs_seas_rem = np.array([])
        valid_months_obs_sel = SIE_obs_trim['V_mon (valid date month)']
        for io_SIE in np.arange(0,len(SIE_obs_trim)):
            mon_io = valid_months_obs_sel.iloc[io_SIE]
            all_mon_io = np.where(valid_months_obs_sel == mon_io)
            io_ice = SIE_obs_trim['SIE'].iloc[io_SIE] - SIE_obs_trim['SIE'].iloc[all_mon_io].mean()
            io_d_ice = SIE_obs_trim['d_SIC (V - I)'].iloc[io_SIE] - SIE_obs_trim['d_SIC (V - I)'].iloc[all_mon_io].mean()
            SIE_obs_seas_rem = np.append(SIE_obs_seas_rem,io_ice)
            d_SIE_obs_seas_rem = np.append(d_SIE_obs_seas_rem,io_d_ice)
        
        SIE_obs_trim['SIE (seas rem)'] = SIE_obs_seas_rem
        SIE_obs_trim['d_SIE (seas rem)'] = d_SIE_obs_seas_rem
        SIE_obs_trim.head()
        # In[32]:
        
        
        #print(SIE_lead_sel['SIE'].mean())
        #print(np.nanmean(signal.detrend(SIE_lead_sel['SIE'])))
        
        
        # Now we have selected data based on region, season, and lead time.  We'll 
        
        # In[33]:
        
        
        plt.plot(SIE_obs_trim['V (valid date)'],SIE_obs_trim['SIE']-SIE_obs_trim['SIE'].mean(),linewidth=2)
        plt.plot(SIE_obs_trim['V (valid date)'],SIE_obs_trim['SIE (seas rem)'],color='r',linewidth=2)
        #plt.plot(SIE_lead_sel['V (valid date)'],signal.detrend(SIE_lead_sel['SIE']),color='g',linewidth=2)
        plt.ylabel('Sea ice extent (10^6 km^2)')
        plt.xticks(SIE_obs_trim['V (valid date)'].iloc[np.arange(0,len(SIE_obs_trim['SIE']),200)],rotation=45)
        #plt.title('Sea ice extent, {region}, {season}, lead time: {lead} days'.format(region=region_name_test,season=month_sel_str,
        #                                                              lead=lead_days_select[lead_days_ind]))
        plt.title('Sea ice extent, {region}, {season}, observations'.format(region=region_name_test,season=month_sel_str))
        
        plt.figure()
        plt.plot(SIE_obs_trim['V (valid date)'],SIE_obs_trim['d_SIC (V - I)'],linewidth=2,color='b')
        plt.plot(SIE_obs_trim['V (valid date)'],SIE_obs_trim['d_SIE (seas rem)'],linewidth=2,color='r')
        plt.ylabel('Change in sea ice extent (10^6 km^2)')
        plt.ylim(-0.25,0.25)
        plt.xticks(SIE_obs_trim['V (valid date)'].iloc[np.arange(0,len(SIE_obs_trim['SIE']),200)],rotation=45)
        #plt.title('Change in sea ice extent, {region}, {season}, lead time: {lead} days'.format(region=region_name_test,season=month_sel_str,
        #                                                              lead=lead_days_select[lead_days_ind]))
        plt.title('Change in sea ice extent, {region}, {season}, observations'.format(region=region_name_test,season=month_sel_str))
        
        # Make histograms using kernel density estimates.  We'll create one for each ensemble separately.  Make a basic plot here. 
        
        # In[52]:
        
        
        SIE_group_ens = SIE_lead_sel.groupby(['ensemble'])
        no_ens = 25
        no_bins = 20
        kde_to_plot = np.empty((no_bins,0))
        d_kde_to_plot = np.empty((no_bins,0))
        
        d_ks_ALL = np.empty((no_bins,0))
        ks_ALL = np.empty((no_bins,0))
        d_p_ALL = np.empty((no_bins,0))
        p_ALL = np.empty((no_bins,0))
        for iens in np.arange(0,no_ens):
            ens_sel = iens + 1
            #SIE_ens_sel = SIE_group_ens['SIE'].get_group(ens_sel)
            SIE_ens_sel = SIE_group_ens['SIE (seas rem)'].get_group(ens_sel)
            d_SIE_ens_sel = SIE_group_ens['d_SIE (seas rem)'].get_group(ens_sel)
            if SIE_ens_sel.mean() == 0:
                continue
            ikde = stats.gaussian_kde(SIE_ens_sel)
            d_ikde = stats.gaussian_kde(d_SIE_ens_sel)
            bin_lims = 1.5*max(abs(SIE_ens_sel.min()),abs(SIE_ens_sel.max()))
            d_bin_lims = 1.85*max(abs(d_SIE_ens_sel.min()),abs(d_SIE_ens_sel.max()))
            #print(bin_lims)
            eval_range = np.linspace(-bin_lims,bin_lims,no_bins)
            d_eval_range = np.linspace(-d_bin_lims,d_bin_lims,no_bins)
            kde_plot = ikde.evaluate(eval_range)
            d_kde_plot = d_ikde.evaluate(d_eval_range)
            kde_to_plot = np.append(kde_to_plot,np.expand_dims(kde_plot,axis=1),axis=1)
            d_kde_to_plot = np.append(d_kde_to_plot,np.expand_dims(d_kde_plot,axis=1),axis=1)
            
            d_ks_ens,d_p_ens = stats.ks_2samp(SIE_obs_trim['d_SIE (seas rem)'],d_SIE_ens_sel)
            ks_ens,p_ens = stats.ks_2samp(SIE_obs_trim['SIE (seas rem)'],SIE_ens_sel)
            d_ks_ALL = np.append(d_ks_ALL,d_ks_ens)
            ks_ALL = np.append(ks_ALL,ks_ens)
            d_p_ALL = np.append(d_p_ALL,d_p_ens)
            p_ALL = np.append(p_ALL,p_ens)
            
        ks_FULL,p_FULL = stats.ks_2samp(SIE_obs_trim['SIE (seas rem)'],SIE_lead_sel['SIE (seas rem)'])
        d_ks_FULL,d_p_FULL = stats.ks_2samp(SIE_obs_trim['d_SIE (seas rem)'],SIE_lead_sel['d_SIE (seas rem)'])
        alpha = 0.025 #two-sided
        id_failures = np.where(p_ALL >= alpha)
        id_d_failures = np.where(d_p_ALL >= alpha)
        no_failures = len(id_failures[0])
        d_no_failures = len(id_d_failures[0])
        
        df_vec = (ireg)*len(lead_days_select) + ilead
        ks_failures.iloc[df_vec,1] = region_name_test
        ks_failures.iloc[df_vec,2] = no_failures
        ks_failures.iloc[df_vec,3] = d_no_failures
        if not kde_to_plot.any():
            print('nothing to plot')
            continue
        
        # Do KDE estimate for observations
        obs_kde = stats.gaussian_kde(SIE_obs_trim['SIE (seas rem)'])
        obs_kde_d = stats.gaussian_kde(SIE_obs_trim['d_SIE (seas rem)'])
        obs_bins = 1.5*max(abs(SIE_obs_trim['SIE (seas rem)'].min()),abs(SIE_obs_trim['SIE (seas rem)'].max()))
        obs_bins_d = 1.25*max(abs(SIE_obs_trim['d_SIE (seas rem)'].min()),abs(SIE_obs_trim['d_SIE (seas rem)'].max()))
        eval_range_obs = np.linspace(-obs_bins,obs_bins,no_bins)
        d_eval_range_obs = np.linspace(-obs_bins_d,obs_bins_d,no_bins)
        
        obs_kde_plot = obs_kde.evaluate(eval_range_obs)
        obs_d_kde_plot = obs_kde_d.evaluate(d_eval_range_obs)
        #
        plt.figure()
        #plt.plot(eval_range,kde_to_plot/no_bins)
        plt.fill_between(eval_range,np.amin(kde_to_plot,axis=1)/no_bins,np.amax(kde_to_plot,axis=1)/no_bins,color='xkcd:salmon')
        plt.plot(eval_range,np.nanmean(kde_to_plot,axis=1)/no_bins,color='k')
        plt.plot(eval_range_obs,obs_kde_plot/no_bins,color='b')
        plt.xlabel('sea ice extent (10^6 km^2)')
        plt.ylabel('Frequency')
        plt.legend(['model','obs'])
        plt.title('Sea ice extent, {region}, {seas}, {lead} days'.format(region=region_name_test,seas=month_sel_str,
                  lead=lead_days_select[ilead]))
        plt.text(eval_range_obs[1],0.02,'failed k-s test {n} times'.format(n=no_failures))
        fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE_v2/histograms/compare_obs_and_model/seas/'
        fname_1 = fpath_save+'SIE_{region}_{seas}_{lead}day_lead_compare_model_{model_name}_{model_type}_and_obs.png'.format(region=region_name_test,
                                  seas=month_sel_str,lead=lead_days_select[ilead],model_name=model_name,model_type=model_type)
        plt.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
        #plt.figure()
        #plt.plot(d_eval_range,d_kde_to_plot)
        ##
        plt.figure()
        #plt.plot(eval_range,kde_to_plot/no_bins)
        plt.fill_between(d_eval_range,np.amin(d_kde_to_plot,axis=1)/no_bins,np.amax(d_kde_to_plot,axis=1)/no_bins,color='xkcd:salmon')
        plt.plot(d_eval_range,np.nanmean(d_kde_to_plot,axis=1)/no_bins,color='k')
        plt.plot(d_eval_range_obs,obs_d_kde_plot/no_bins,color='b')
        plt.xlabel('change in sea ice extent (10^6 km^2)')
        plt.ylabel('Frequency')
        plt.legend(['model','obs'])
        plt.title('{no_days}-day change in sea ice extent, {region}, {seas}, {lead} days'.format(no_days=no_day_change,
                  region=region_name_test,seas=month_sel_str,lead=lead_days_select[ilead]))
        plt.text(d_eval_range_obs[1],0.2,'failed k-s test {n} times'.format(n=d_no_failures))
        fname_2 = fpath_save+'d_SIE_{no_day}day_change_{region}_{seas}_{lead}day_lead_compare_model_{model_name}_{model_type}_and_obs.png'.format(no_day=no_day_change,
                region=region_name_test,seas=month_sel_str,lead=lead_days_select[ilead],model_name=model_name,model_type=model_type)
        plt.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')
        plt.close('all')

# In[ ]:




