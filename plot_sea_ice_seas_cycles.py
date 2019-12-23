#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

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
    
filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/'
model_name = 'ecmwfsipn'
model_type = 'reforecast'
day_change = 5

filename = filepath+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                       model_type=model_type,d_days=day_change)


# Load the SIE dataframe

# In[2]:


ds_SIC_all = pd.read_csv(filename)
#ds_SIC_all = ds_SIC_all.drop(columns='lead time (V - I)')
ds_SIC_all = ds_SIC_all.dropna()
regions = ds_SIC_all['region'].unique().tolist() #list unique regions
##remove St John, Hudson Bay, Sea of Okhostk because they fuck stuff up
#regions.remove('St John')
#regions.remove('Hudson Bay')
#regions.remove('Bering')
#regions.remove('Sea of Okhotsk')
lead_days = ds_SIC_all['lead time (days)'].unique().tolist()
#print(lead_days)
ds_SIC_all.head()
#toggle for SIE or change in SIE
choose_var_model = 'SIE'
choose_var_obs = 'SIE'
titlename_choose_var = 'Sea ice extent'.format(day_change=day_change)
filename_choose_var = 'SIE'
# Initialize dataframe to keep track of bias.  Bias is defined as Model - Obs.  So a negative value means model SIE is less than observed; positive value means model SIE is greater than observed. 

# In[3]:


mons_vec = np.arange(1,13)
mon_tile = np.tile(mons_vec,len(regions))
df_bias = pd.DataFrame(columns=['SIE bias','month','region'])
df_bias['month'] = mon_tile

fpath_save_im = '/home/disk/sipn/mcmcgraw/figures/VRILE_v2/sea_ice_seas_cycle/'
# Group by region, then lead time--we'll plot the seasonal cycle in SIE as a function of lead time.  Start w/ 2 days

#%% Load obs
obs_name = 'NSIDC_0079'
obs_type = 'sipn_nc_yearly_agg'
obs_filename = filepath+'{model_name}_{model_type}_SIE_d_SIE_{d_days}day_change_lead_time_ALL_REGIONS_ALL_ENS.csv'.format(model_name=obs_name,
                       model_type=obs_type,d_days=day_change)
SIE_obs = pd.read_csv(obs_filename)
SIE_obs = SIE_obs.dropna()
SIE_obs.head()
obs_yr = pd.to_datetime(SIE_obs['V (valid date)']).dt.year
yrmin = 1993
yrmax = 2016
yr_vec = np.arange(yrmin,yrmax)
f_yr_obs = np.isin(obs_yr,yr_vec) 
#SIE_obs_trim = SIE_obs.loc[f_yr_obs]
# In[4]:

for ireg in np.arange(0,len(regions)):
    reg_ind_sel = ireg
    region_sel = regions[reg_ind_sel]
    bias_save_ind = reg_ind_sel*12 + np.arange(0,12)
    df_bias.loc[bias_save_ind,'region'] = region_sel
    #print(df_bias)
    print('running region {region}'.format(region=region_sel))
    SIC_reg_group = ds_SIC_all.groupby(['region'])
    SIC_reg_sel = SIC_reg_group.get_group(region_sel)
    #Now group by lead time
    lead_ind_sel = 2
    SIC_lead_group = SIC_reg_sel.groupby(['lead time (days)'])
    SIC_lead_sel = SIC_lead_group.get_group(lead_ind_sel)
    
    
    # Now group by valid date to create a seasonal cycle
    
    # In[5]:
    
    
    valid_dates = pd.to_datetime(SIC_lead_sel['V (valid date)'])
    valid_dates_month = pd.to_datetime(SIC_lead_sel['V (valid date)']).dt.month
    SIC_mon_sel = SIC_lead_sel
    SIC_mon_sel['valid date month'] = valid_dates_month
    
    
    # Group by month of valid date and make a box plot
    
    # In[6]:
    
    
    import seaborn as sns
    
    SIC_mon_group = SIC_mon_sel.groupby(['valid date month'])
    fig1 = plt.figure(1)
    ax1=sns.boxplot(x='valid date month',y=choose_var_model,data=SIC_mon_sel)
    ax1.set_xlabel('Month',fontsize=13)
    ax1.set_ylabel('Sea Ice Extent ($10^6 km^2$)',fontsize=13)
    ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax1.set_title('{title_var}, {region} ({model_name} {model_type}), lead days: {lead_day}'.format(title_var=titlename_choose_var,
                  region=region_sel,
                  model_name=model_name,model_type=model_type,lead_day=lead_ind_sel),fontsize=14)
    fname_save1 = fpath_save_im+'{filename_choose_var}_seas_cycle_{region}_{model_name}_{model_type}_{lead_day}.png'.format(filename_choose_var=filename_choose_var,
                                       region=region_sel,
                                             model_name=model_name,model_type=model_type,lead_day=lead_ind_sel)
    fig1.savefig(fname_save1,format='png',dpi=600,bbox_inches='tight')
    # Add observations to plot
    
    # In[7]:

    # Group by region and then month
    
    # In[8]:
    
    
    SIE_obs_reg = SIE_obs.groupby(['region'])
    SIE_obs_reg_sel = SIE_obs_reg.get_group(region_sel)
    print('running region {region}'.format(region=region_sel))
    SIE_obs_dates = pd.to_datetime(SIE_obs_reg_sel['V (valid date)'])
    f_dates = SIE_obs_dates.isin(valid_dates)
    SIE_obs_trim = SIE_obs_reg_sel.loc[f_dates]
    SIE_obs_mon = SIE_obs_trim.groupby(['V_mon (valid date month)'])
    SIE_obs_mon_mean = SIE_obs_mon.median()
    stat = 'median'
    #print(SIE_obs_mon_mean)
    SIE_seas = SIE_obs_mon_mean[choose_var_obs]
    
    
    # Make plot with model and obs
    
    # In[9]:
    
    fig2 = plt.figure(2)
    ax2=sns.boxplot(x='valid date month',y=choose_var_model,data=SIC_mon_sel)
    ax2.set_xlabel('Month',fontsize=13)
    ax2.set_ylabel('{title_var} Sea Ice Extent (10^6 km^2)'.format(title_var=titlename_choose_var),fontsize=13)
    pto = ax2.plot(np.arange(0,12),SIE_seas,'rx',markersize=10,markeredgewidth=3)
    
    
    # Try a different kind of plot--box plots side-by-side.  Easiest way is to create a new dataframe so we can use seaborn side by side
    
    # In[61]:
    
    
    model_double_plot = pd.DataFrame(columns=['SIE','month','model?'])
    model_double_plot['SIE'] = SIC_mon_sel[choose_var_model]
    model_double_plot['month'] = SIC_mon_sel['valid date month']
    model_double_plot['model?'] = 'model'
    
    obs_double_plot = pd.DataFrame(columns=['SIE','month','model?'])
    obs_double_plot['SIE'] = SIE_obs_reg_sel[choose_var_obs]
    obs_double_plot['month'] = SIE_obs_reg_sel['V_mon (valid date month)'].astype(np.int64)
    obs_double_plot['model?'] = 'obs'
    
    df_double_plot = pd.concat([model_double_plot,obs_double_plot],axis=0)
    fig5 = plt.figure(5)
    ax5 = sns.boxplot(x='month',y='SIE',hue='model?',data=df_double_plot)
    ax5.set_xlabel('Month',fontsize=13)
    ax5.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontsize=12)
    ax5.set_ylabel('Sea ice extent ($10^6 km^2$)',fontsize=13)
    ax5.set_title('{titlename} sea ice extent, {region}, lead days: {lead_day}'.format(titlename=titlename_choose_var,
                  region=region_sel,lead_day=lead_ind_sel),fontsize=14)
    fname_save5 = fpath_save_im+'{filename_choose}_seas_cycle_{region}_OBS_and_{model_name}_{model_type}_{lead_day}.png'.format(filename_choose=filename_choose_var,
            region=region_sel,model_name=model_name,model_type=model_type,lead_day=lead_ind_sel)
    fig5.savefig(fname_save5,format='png',dpi=600,bbox_inches='tight')
    
    # Plot differences between obs mean and model mean
    
    # In[62]:
    
    
    fig3 = plt.figure(3)
    ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])
    ax3.plot(np.arange(0,12),(SIC_mon_group[choose_var_model].median()-SIE_seas),'ko--',markersize=8)
    ax3.set_xlabel('Month',fontsize=13)
    ax3.set_xticks(ticks=np.arange(0,13))
    ax3.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax3.set_ylabel('Sea ice extent $(10^6 km^2)$',fontsize=13)
    ax3.set_title('Bias (model - obs) in {stat} {titlename} sea ice extent, {region}, lead days: {lead_day}'.format(titlename=titlename_choose_var,
            stat=stat,region=region_sel,lead_day=lead_ind_sel),fontsize=14)
    fname_save3 = fpath_save_im+'{filename}_seas_cycle_BIAS_{stat}_{region}_{model_name}_{model_type}_{lead_day}.png'.format(filename=filename_choose_var,
                                                     stat=stat,region=region_sel,model_name=model_name,model_type=model_type,lead_day=lead_ind_sel)
    fig3.savefig(fname_save3,format='png',dpi=600,bbox_inches='tight')
    
    # In[63]:
    
    
    bias = SIC_mon_group[choose_var_model].median()-SIE_seas
    #Replace zeros with NaNs
    bias_adj = bias
    bias_adj.loc[bias==0] = np.nan
    df_bias.loc[bias_save_ind,'SIE bias'] = bias_adj.values
    df_bias.loc[bias_save_ind,'SIE bias'] = bias.values
    
    plt.close('all')


# In[ ]: Make a heatmap for the bias.

df_bias_plot = df_bias.pivot(index='region',columns='month',values='SIE bias')
df_bias_plt = df_bias_plot.astype('float')
fig6 = plt.figure(6)
vmin = -1
vmax = 1
ax6 = sns.heatmap(df_bias_plt,vmin=vmin,vmax=vmax,cmap='seismic',linewidths=1,
                  linecolor='k',cbar_kws={'label': '10^6 km^2',
                                          'ticks':np.linspace(vmin,vmax,9)})
cbar = ax6.collections[0].colorbar
cbar.set_label('$10^6 km^2$',fontsize=11,rotation=0,y=-0.015,labelpad=-15)
ax6.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax6.set_xlabel('Month',fontsize=13)
ax6.set_ylabel('Region',fontsize=13)
ax6.set_title('Bias in {stat} {titlename} Sea Ice Extent (Model - Obs), lead days: {lead_day}'.format(titlename=titlename_choose_var,
        stat=stat,lead_day=lead_ind_sel),fontsize=14)
fname_save6 = fpath_save_im+'BIAS_{stat}_{filename}_all_regions_all_mon_{model_name}_{model_type}_{lead_day}.png'.format(filename=filename_choose_var,
                                  stat=stat,region=region_sel,
                                  model_name=model_name,model_type=model_type,lead_day=lead_ind_sel)
fig6.savefig(fname_save6,format='png',dpi=600,bbox_inches='tight')
#fname_save = 


df_bias_lim = df_bias_plt.drop(index=['St John','Sea of Okhotsk','Beaufort Sea',
                                      'Bering','Canadian Islands','Chukchi Sea',
                                      'East Siberian Sea','Kara Sea','Hudson Bay',
                                      'Laptev Sea'])
#  
fig7 = plt.figure(7)
ax7 = sns.heatmap(df_bias_lim,vmin=vmin,vmax=vmax,cmap='seismic',linewidths=1,
                  linecolor='k',cbar_kws={'label': '10^6 km^2',
                                          'ticks':np.linspace(vmin,vmax,9)})
cbar7 = ax7.collections[0].colorbar
cbar7.set_label('$10^6 km^2$',fontsize=11,rotation=0,y=-0.0185,labelpad=-15)
ax7.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax7.set_xlabel('Month',fontsize=13)
ax7.set_ylabel('Region',fontsize=13)
ax7.set_title('Bias in {stat} {titlename} Sea Ice Extent (Model - Obs), lead days: {lead_day}'.format(titlename=titlename_choose_var,
        stat=stat,lead_day=lead_ind_sel),fontsize=14)
fname_save7 = fpath_save_im+'BIAS_{stat}_{filename}_LIM_regions_all_mon_{model_name}_{model_type}_{lead_day}.png'.format(filename=filename_choose_var,
        stat=stat,region=region_sel,model_name=model_name,model_type=model_type,lead_day=lead_ind_sel)
fig7.savefig(fname_save7,format='png',dpi=600,bbox_inches='tight')
#%%
#Line plot
#df_bias_lim = df_bias_plt
fig8 = plt.figure(8)
ax8 = fig8.add_axes([0.1,0.1,0.9,0.9])
ax8.plot(np.transpose(df_bias_lim),'o-',linewidth=1.5,markersize=6)
ax8.grid()
ax8.axhline(y=0,color='k',linewidth=1.5)
xlabels = df_bias_lim.index
ax8.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
ax8.set_xticklabels(labels=['J','F','M','A','M','J','J','A','S','O','N','D'],rotation=0)
ax8.legend(xlabels,loc='best',bbox_to_anchor=(1.525, 0.5, 0., 0),fontsize=10)
ax8.set_xlabel('Month',fontsize=13)
ax8.set_ylabel('$10^6 km^2$',fontsize=13)
ax8.set_title('Bias in {stat} {titlename} Sea Ice Extent (Model - Obs), lead days: {lead_day}'.format(titlename=titlename_choose_var,
              stat=stat,lead_day=lead_ind_sel),fontsize=14)
fname_save8 = fpath_save_im+'BIAS_{stat}_{filename}_LIM_line_plot_regions_all_mon_{model_name}_{model_type}_LEAD_DAY_{lead_day}.png'.format(filename=filename_choose_var,
              stat=stat,region=region_sel,model_name=model_name,model_type=model_type,lead_day=lead_ind_sel)
fig8.savefig(fname_save8,format='png',dpi=600,bbox_inches='tight')