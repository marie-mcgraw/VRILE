#!/usr/bin/env python
# coding: utf-8

# In[72]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import pandas as pd
import seaborn as sns


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
#SIE_obs.head()
# Make histograms grouped by region, lead day, and valid date. Start with regions.  Use groupby to group dataframe by regions, and unique() to get the list of regions

# In[3]:


region_names = SIE_data['region'].dropna().unique().tolist()
#region_ind_test = 15
for ireg in np.arange(0,len(region_names)):
    region_ind_test = ireg
    print('now running {region}'.format(region=region_names[region_ind_test]))
    region_name_test = region_names[region_ind_test]
    SIE_region_groupby = SIE_data.groupby(['region'])
    SIE_by_region = SIE_region_groupby.get_group(region_name_test)
    SIEobs_region_groupby = SIE_obs.groupby(['region'])
    SIEobs_by_region = SIEobs_region_groupby.get_group(region_name_test)
    
    # Now select for month of valid date (JJAS for now)
    
    # In[4]:
    
    
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
    
    # Now, filter by lead date.  We'll start with lead times of 1-4 weeks.  Then we remove seasonality. We'll just do it by month--remove mean of all June days from each June day, etc. Add a new column to the existing dataframe with seasonal anomalies of SIE, and one with seasonal anomalies of the change in SIE
    
    # In[110]:
    
    
    #f_lead = SIE_month_trim.where(SIE_month_trim['lead time (days)'] == 7)
    SIE_lead_groups = SIE_month_trim.groupby(['lead time (days)'])
    SIE_month_trim['SIE (seas rem)'] = ""
    SIE_month_trim['d_SIE (seas rem)'] = ""
    lead_days_select = [7,14,21,28,35,42,49,56]
    for ilead in np.arange(0,len(lead_days_select)):
        SIE_lead_sel = SIE_lead_groups.get_group(lead_days_select[ilead])
        is_detrend = False
        if is_detrend == True:
            SIE_dt = signal.detrend(SIE_lead_sel['SIE'])
            SIE_lead_sel['SIE'] = SIE_dt
            SIEobs_dt = signal.detrend(SIE_obs_trim['SIE'])
            SIE_obs_trim['SIE'] = SIEobs_dt
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
        #print(SIE_seas_rem)
        #print(d_SIE_seas_rem)
        SIE_lead_sel['SIE (seas rem)'] = SIE_seas_rem
        SIE_lead_sel['d_SIE (seas rem)'] = d_SIE_seas_rem
        f_lead = SIE_month_trim.index[SIE_month_trim['lead time (days)'] == lead_days_select[ilead]]
        SIE_month_trim['SIE (seas rem)'].loc[f_lead] = SIE_seas_rem
        SIE_month_trim['d_SIE (seas rem)'].loc[f_lead] = d_SIE_seas_rem
        
        #SIE_lead_sel.head()
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
    #plt.plot(SIE_lead_sel['SIE'])
    #plt.plot(SIE_lead_sel['SIE'],color='r')
    
    
    # Make box plots as a function of lead time 
    
    # In[120]:
    
    
    f_lead_trim = SIE_month_trim['lead time (days)'].isin(lead_days_select)
    SIE_lead_trim = SIE_month_trim.loc[f_lead_trim]
    SIE_lead_trim_groups = SIE_lead_trim.groupby(['lead time (days)'])
    #Box plot properties
    medianprops = {'color': 'blue', 'linewidth': 2}
    boxprops = {'linestyle':'-', 'linewidth': 3}
    whiskerprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    capprops = {'color': 'black', 'linestyle': '-', 'linewidth': 3}
    flierprops = {'color': 'black', 'marker': 'o', 'markersize': 1, 'markeredgewidth': 2}
    #    
    fig1 = plt.figure(1)
    for iplt in np.arange(0,len(lead_days_select)):
        ax1 = sns.boxplot(x=SIE_lead_trim['lead time (days)'].dropna(),y=SIE_lead_trim['SIE (seas rem)'].astype(float),
                     medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops,
                          capprops=capprops,flierprops=flierprops)
    ax1.set_ylabel('Sea ice extent (10^6 km^2)')
    ax1.set_title('Sea ice extent, {seas}, {region}'.format(seas=month_sel_str,region=region_name_test))
    
    fpath_save = '/home/disk/sipn/mcmcgraw/figures/VRILE_v2/boxplots/vs_lead_time/seas/'
    fname_1 = fpath_save+'SIE_{region}_{seas}_{lead}day_lead_{model_name}_{model_type}.png'.format(region=region_name_test,
                                      seas=month_sel_str,lead=lead_days_select[ilead],model_name=model_name,model_type=model_type)
    fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
    # In[121]:
    
    
    fig2 = plt.figure(2)
    for iplt in np.arange(0,len(lead_days_select)):
        ax2 = sns.boxplot(x=SIE_lead_trim['lead time (days)'],y=SIE_lead_trim['d_SIE (seas rem)'].astype(float),
                     medianprops=medianprops,boxprops=boxprops,whiskerprops=whiskerprops,
                          capprops=capprops,flierprops=flierprops)
    ax2.set_ylabel('Sea ice extent (10^6 km^2)')
    ax2.set_title('{day}-day change in sea ice extent, {seas}, {region}'.format(day=no_day_change,seas=month_sel_str,
                                                                                   region=region_name_test))
    fname_2 = fpath_save+'d_SIE_{region}_{seas}_{lead}day_lead_{model_name}_{model_type}.png'.format(region=region_name_test,
                                      seas=month_sel_str,lead=lead_days_select[ilead],model_name=model_name,model_type=model_type)
    fig2.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')
    
    plt.close('all')

# In[ ]:




