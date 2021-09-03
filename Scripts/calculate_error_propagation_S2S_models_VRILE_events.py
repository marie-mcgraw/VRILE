#!/usr/bin/env python
# coding: utf-8

# <code>calculate_error_propagation_S2S_models_VRILE_days_EVENTS_with_LOO.ipynb</code>.  We calculate the error propagation that occurs during/after VRILE days by analyzing forecasts that start a certain number of days before a VRILE event.

# In[1]:


import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from S2S_sea_ice_preprocess import load_model,create_aggregate_regions,create_model_climatology,create_obs_climatology
from S2S_sea_ice_VRILEs import get_VRILE_days_EVENTS
from S2S_sea_ice_metrics import calculate_errors,get_pvalues


# <b>inputs:</b><br>
# <li>  model name (ecmwf,ukmo,ncep,metreofr) </li>
# <li>  seas_str [string for season; ALL if we want to do full year]</li>
# <li>  seas_sel [months of season; empty if we want to do full year] </li>
# <li>  vrile_thresh [threshhold at which VRILE is estimated </li>
# <li>  thresh_str [string for VRILE threshhold] </li>
# 

# In[2]:


model_name_ALL = ['ukmo','ecmwf','metreofr','ncep']
seas_str_ALL = ['JAS','JJA','JFM','ALL','JJAS','DJFM','DJF','AMJ','OND'] # ALL
seas_sel_ALL = [[7,8,9],[6,7,8],[1,2,3],[1,2,3,4,5,6,7,8,9,10,11,12],
                [6,7,8,9],[1,2,3,12],[1,2,12],[4,5,6],[10,11,12]] #[1,2,3,4,5,6,7,8,9,10,11,12]
obs_name = 'NSIDC_0079'
vrile_thresh_ALL = [0.05,0.1]
thresh_str_ALL = ['05','10']
WEEKLY = True
lead_weeks = True
nday_change = 5 #number of days for VRILE calculation
normalize = False
VRILE_shift = 14 # days; number of days BEFORE VRILE to analyze
COMMON_RF = True
max_date_offset = 5 # days; number of days +/- the start of the VRILE
drop_last = True


# Load model output for our desired model

# In[3]:
for i_thresh in np.arange(0,len(vrile_thresh_ALL)):
    vrile_thresh = vrile_thresh_ALL[i_thresh]
    thresh_str = thresh_str_ALL[i_thresh]
    print('VRILE thresh is ',thresh_str)
    #
    for i_seas in np.arange(0,len(seas_str_ALL)):
        seas_str = seas_str_ALL[i_seas]
        seas_sel = seas_sel_ALL[i_seas]
        print('season is ',seas_str)
        #
        for model_name in model_name_ALL:
            print('running model ',model_name)

            SIE = load_model(model_name)
            print('loaded ',model_name)


            # Create aggregate regions that combine some of the NSIDC-MASIE regions

            # In[4]:


            SIE = create_aggregate_regions(SIE)
            print('combined regions')


            # Take ensemble mean, get lead time in days, and convert to Dataframe

            # In[5]:


            SIE_ens_mean = SIE.mean(dim='ensemble')
            regions = SIE.region_names
            lead_days = SIE.fore_time.dt.days
            SIE_df = SIE_ens_mean.to_dataframe().reset_index()


            # Calculate the date for forecasts by adding the <code>fore_time</code> to <code>init_time</code>. Rename some columns to make life easier

            # In[6]:


            SIE_df['valid date'] = SIE_df['init_time'] + SIE_df['fore_time']
            SIE_df = SIE_df.rename(columns={'region_names':'region',
                                       'fore_time':'lead time (days)',
                                       'init_time':'init date',
                                       'Extent':'SIE'})


            # Create climatology for model output.  Decide how long we want weeks to be for weekly climatology (default is 7 days)

            # In[7]:


            week_length = 7
            SIE_df = create_model_climatology(SIE_df,7)
            print('model climatology created')


            # Load observations.  NSIDC_0079 is NASA Bootstrap, NSIDC_0051 is NASA team

            # In[8]:


            if model_name == 'NSIDC_0051':
                obs_type = 'sipn_nc_yearly_agg'
            else:
                obs_type = 'sipn_nc_yearly_agg_commonland'
            filepath = '/home/disk/sipn/nicway/data/obs/{model_name}/{model_type}/'.format(model_name=obs_name,
                                                                                          model_type=obs_type)
            obs_filenames = xr.open_mfdataset(filepath+'/*.nc',combine='by_coords')
            print('opening ',obs_filenames)
            obs_SIE = obs_filenames.Extent
            obs_regions = obs_filenames.nregions
            obs_region_names = obs_filenames['region_names'].values
            # Drop region names and re-add as a non-dask.array object.  This is stupid but oh well
            obs_SIE = obs_SIE.drop('region_names')
            obs_SIE["region_names"] = ("nregions",obs_region_names)
            print('obs loaded')


            # Add aggregate regions to obs and convert obs to Pandas dataframe

            # In[9]:


            obs_SIE = create_aggregate_regions(obs_SIE)
            obs_SIE = obs_SIE.to_dataframe().reset_index()
            obs_SIE = obs_SIE.rename(columns={'Extent':'SIE','region_names':'region','time':'valid date'})
            obs_SIE['valid year'] = pd.to_datetime(obs_SIE['valid date']).dt.year


            # Calculate our observed climatology 

            # In[10]:


            if COMMON_RF == True:
                obs_SIE = obs_SIE[pd.to_datetime(obs_SIE['valid date']).dt.year.isin(np.arange(1999,2015))]
                obs_SIE = create_obs_climatology(obs_SIE)
                time_str = 'COMMON_RF'
                print('common reforecast')
            else:
                time_str = 'FULL_PERIOD'
                obs_SIE = create_obs_climatology(obs_SIE)
                print('full period')
            print('observed climatology created')


            # Now calculate RMSE based on VRILE events--we want to identify forecasts that start some number of days BEFORE the first day of a VRILE, and watch how the RMSE evolves.  So first, we need to get VRILE days and then identify forecasts that start $n$ days before.  We also need to track VRILE EVENTS--that is, if consecutive days are VRILE days, they are part of the same EVENT.  EVENTS must be separated by <code>BUFFER_DAYS</code> days to be considered separate events.

            # In[11]:


            import warnings;
            warnings.filterwarnings('ignore');


            # Initialize dataframes

            # In[12]:


            SIE_errors_ALL = pd.DataFrame()
            SIE_anom_errors_ALL = pd.DataFrame()
            SIE_errors_NO_ALL = pd.DataFrame()
            SIE_anom_errors_NO_ALL = pd.DataFrame()
            #
            SIE_VRILES_TEST = pd.DataFrame()
            obs_VRILES_TEST = pd.DataFrame()
            SIE_no_VRILES_TEST = pd.DataFrame()
            obs_no_VRILES_TEST = pd.DataFrame()
            SIE_anom_VRILES_TEST = pd.DataFrame()
            obs_anom_VRILES_TEST = pd.DataFrame()
            SIE_anom_no_VRILES_TEST = pd.DataFrame()
            obs_anom_no_VRILES_TEST = pd.DataFrame()
            #
            SIE_reg = SIE_df.set_index(['region'])
            regions_list = SIE_df['region'].unique().tolist()
            #
            pvalues_SIE = pd.DataFrame()
            pvalues_SIE_anom = pd.DataFrame()
            buffer_days = 14
            yrs = obs_SIE['valid year'].unique().tolist()
            #
            week_length = 7
            if (model_name != 'ukmo') & (drop_last == True):
                max_fore = SIE_reg['lead time (days)'].max()
                SIE_reg = SIE_reg.where(SIE_reg['lead time (days)'] < max_fore).dropna(how='all')
            #SIE_reg.groupby(['region','lead time (days)'])['SIE'].mean().xs('East Siberian-Beaufort-Chukchi Sea')#.plot()


            # Here we go!

            # In[13]:


            for iyr in yrs:
                obs_SIE_sel = obs_SIE[~obs_SIE['valid year'].isin([iyr])]
                print('leaving out ',iyr)
                # Estimate observed VRILE days
                obs_SIE_VRILE_onlyx, obs_SIE_anom_VRILE_onlyx, obs_SIE_NO_VRILEx, obs_SIE_anom_NO_VRILEx = get_VRILE_days_EVENTS(obs_SIE_sel,vrile_thresh,nday_change,seas_sel,buffer_days)
                print('VRILE days calculated')
                #
                # Now, we want to know how well the models forecast ONLY those VRILE days. 
                obs_SIE_VRILE_only = obs_SIE_VRILE_onlyx.set_index(['region'])
                obs_SIE_anom_VRILE_only = obs_SIE_anom_VRILE_onlyx.set_index(['region'])
                #
                obs_SIE_NO_VRILE = obs_SIE_NO_VRILEx.set_index(['region'])
                obs_SIE_anom_NO_VRILE = obs_SIE_anom_NO_VRILEx.set_index(['region'])
                # Day we START our predictions is determined by VRILE_shift
                obs_SIE_VRILE_only['valid date START'] = obs_SIE_VRILE_only['valid date'] - pd.Timedelta(VRILE_shift,'D')
                obs_SIE_anom_VRILE_only['valid date START'] = obs_SIE_anom_VRILE_only['valid date'] - pd.Timedelta(VRILE_shift,'D')
                # Now, we find model forecasts that start up to max_date_offset days before valid date START
                SIE_df['lead time (weeks)'] = np.floor(SIE_df['lead time (days)'].dt.days/week_length)
                SIE_df_reg = SIE_df.set_index(['region'])
                #x_reg = 'panArctic'
                region_list = obs_SIE_sel['region'].unique().tolist()
                SIE_VRILES = pd.DataFrame()
                obs_VRILES = pd.DataFrame()
                SIE_no_VRILES = pd.DataFrame()
                obs_no_VRILES = pd.DataFrame()
                # Same, but for VRILES based on anomalous SIE
                SIE_anom_VRILES = pd.DataFrame()
                obs_anom_VRILES = pd.DataFrame()
                SIE_anom_no_VRILES = pd.DataFrame()
                obs_anom_no_VRILES = pd.DataFrame()
                for x_reg in region_list:
                    #SIE_df_x = SIE_df.set_index(['region']).xs((x_reg))
                    dates_shifted = pd.DataFrame()
                    dates_shifted['region'] = x_reg
                    dates_shifted = pd.DataFrame(obs_SIE_VRILE_only.xs(x_reg)['valid date START'])
                    dates_shifted_list = pd.DataFrame(obs_SIE_VRILE_only.xs(x_reg)['valid date START'])
                    # Anom dates shifted
                    anom_dates_shifted = pd.DataFrame()
                    anom_dates_shifted['region'] = x_reg
                    anom_dates_shifted = pd.DataFrame(obs_SIE_anom_VRILE_only.xs(x_reg)['valid date START'])
                    anom_dates_shifted_list = pd.DataFrame(obs_SIE_anom_VRILE_only.xs(x_reg)['valid date START'])
                    #
                    for i in np.arange(1,max_date_offset+1):
                        #i_forward = dates_shifted + pd.Timedelta(i,'D')
                        i_backward = dates_shifted - pd.Timedelta(i,'D')
                        dates_shifted_list = dates_shifted_list.append((i_backward))
                        # same but for anom
                        #i_anom_forward = anom_dates_shifted + pd.Timedelta(i,'D')
                        i_anom_backward = anom_dates_shifted - pd.Timedelta(i,'D')
                        anom_dates_shifted_list = anom_dates_shifted_list.append((i_anom_backward)) 
                    #dates_shifted_list_ALL = dates_shifted_list_ALL.append(dates_shifted_list)
                    x_SIE = SIE_df_reg.loc[x_reg]
                    x_SIE_VRILES = x_SIE[x_SIE['init date'].isin(dates_shifted_list['valid date START'])]
                    x_obs = obs_SIE_sel.set_index('region').loc[x_reg]
                    x_SIE_obs = x_obs[x_obs['valid date'].isin(x_SIE_VRILES['valid date'])]
                    SIE_VRILES = SIE_VRILES.append(x_SIE_VRILES)
                    obs_VRILES = obs_VRILES.append(x_SIE_obs)
                    #
                    x_SIE_no = x_SIE[~x_SIE['init date'].isin(dates_shifted_list['valid date START'])]
                    SIE_no_VRILES = SIE_no_VRILES.append(x_SIE_no)
                    x_no_obs = x_obs[x_obs['valid date'].isin(x_SIE_no['valid date'])]
                    obs_no_VRILES = obs_no_VRILES.append(x_no_obs)
                    ### Same, but for anom
                    x_anom_SIE_VRILES = x_SIE[x_SIE['init date'].isin(anom_dates_shifted_list['valid date START'])]
                    #x_anom_obs = obs_SIE.set_index('region').loc[x_reg]
                    x_anom_SIE_obs = x_obs[x_obs['valid date'].isin(x_anom_SIE_VRILES['valid date'])]
                    SIE_anom_VRILES = SIE_anom_VRILES.append(x_anom_SIE_VRILES)
                    obs_anom_VRILES = obs_anom_VRILES.append(x_anom_SIE_obs)
                    #
                    x_anom_SIE_no = x_SIE[~x_SIE['init date'].isin(anom_dates_shifted_list['valid date START'])]
                    SIE_anom_no_VRILES = SIE_anom_no_VRILES.append(x_anom_SIE_no)
                    x_anom_no_obs = x_obs[x_obs['valid date'].isin(x_anom_SIE_no['valid date'])]
                    obs_anom_no_VRILES = obs_anom_no_VRILES.append(x_anom_no_obs)
                    # Calculate RMSE and MAE
                    if x_reg == 'East Siberian-Beaufort-Chukchi Sea':
                        SIE_VRILES_TEST = SIE_VRILES_TEST.append(x_SIE_VRILES)
                        obs_VRILES_TEST = obs_VRILES_TEST.append(x_SIE_obs)
                        SIE_no_VRILES_TEST = SIE_no_VRILES_TEST.append(x_SIE_no)
                        obs_no_VRILES_TEST = obs_no_VRILES_TEST.append(x_no_obs)
                        SIE_anom_VRILES_TEST = SIE_anom_VRILES_TEST.append(x_anom_SIE_VRILES)
                        obs_anom_VRILES_TEST = obs_anom_VRILES_TEST.append(x_anom_SIE_obs)
                        SIE_anom_no_VRILES_TEST = SIE_anom_no_VRILES_TEST.append(x_anom_SIE_no)
                        obs_anom_no_VRILES_TEST = obs_anom_no_VRILES_TEST.append(x_anom_no_obs)
                if lead_weeks == True:
                    clim_freq_str = 'WEEKLY'
                    SIE_VRILES['lead days'] = SIE_VRILES['lead time (weeks)']
                    SIE_anom_VRILES['lead days'] = SIE_anom_VRILES['lead time (weeks)']
                    SIE_raw_err,SIE_errors = calculate_errors(SIE_VRILES.reset_index(),obs_VRILES.reset_index())
                    SIE_anom_raw_err,SIE_anom_errors = calculate_errors(SIE_anom_VRILES.reset_index(),obs_anom_VRILES.reset_index())
                    ## NO VRILES
                    SIE_no_VRILES['lead days'] = SIE_no_VRILES['lead time (weeks)']
                    SIE_anom_no_VRILES['lead days'] = SIE_anom_no_VRILES['lead time (weeks)']
                    SIE_raw_err_NO,SIE_errors_NO = calculate_errors(SIE_no_VRILES.reset_index(),obs_SIE_sel)
                    SIE_anom_raw_err_NO,SIE_anom_errors_NO = calculate_errors(SIE_anom_no_VRILES.reset_index(),
                                                                              obs_anom_no_VRILES.reset_index())
                else:
                    clim_freq_str = 'DAILY'
                    SIE_VRILES['lead days'] = SIE_VRILES['lead time (days)'].dt.days
                    SIE_anom_VRILES['lead days'] = SIE_anom_VRILES['lead time (days)'].dt.days
                    SIE_raw_err,SIE_errors = calculate_errors(SIE_VRILES,obs_shifted_dates)
                    SIE_anom_raw_err,SIE_anom_errors = calculate_errors(SIE_anom_VRILES,obs_anom_shifted_dates)
                    ## NO VRILES
                    SIE_no_VRILES['lead days'] = SIE_no_VRILES['lead time (days)'].dt.days
                    SIE_anom_no_VRILES['lead days'] = SIE_anom_no_VRILES['lead time (days)'].dt.days
                    SIE_raw_err_NO,SIE_errors_NO = calculate_errors(SIE_no_VRILES,obs_no_shifted_dates)
                    SIE_anom_raw_err_NO,SIE_anom_errors_NO = calculate_errors(SIE_anom_no_VRILES,obs_anom_no_shifted_dates)
                print('errors calculated')
                #
                # Get p-values
                sd_VRILE,sd_noVRILE,p_value,N_vrile,N_novrile = get_pvalues(SIE_VRILES,SIE_no_VRILES,SIE_errors,SIE_errors_NO)
                sd_VRILE_anom,sd_noVRILE_anom,p_value_anom,N_vrile_anom,N_novrile_anom = get_pvalues(SIE_anom_VRILES,
                                                                        SIE_anom_no_VRILES,SIE_anom_errors,SIE_anom_errors_NO)
                # 
                # Add information to dataframes
                SIE_errors['year out'] = iyr
                SIE_errors['SIE sdev'] = sd_VRILE
                SIE_errors['sample size'] = N_vrile
                SIE_errors['p-value'] = p_value
                SIE_errors_NO['year out'] = iyr
                SIE_errors_NO['SIE sdev'] = sd_noVRILE
                SIE_errors_NO['sample size'] = N_novrile
                SIE_errors_NO['p-value'] = p_value
                #
                SIE_anom_errors['year out'] = iyr
                SIE_anom_errors['SIE sdev'] = sd_VRILE_anom
                SIE_anom_errors['sample size'] = N_vrile_anom
                SIE_anom_errors['p-value'] = p_value_anom
                SIE_anom_errors_NO['year out'] = iyr
                SIE_anom_errors_NO['SIE sdev'] = sd_noVRILE_anom
                SIE_anom_errors_NO['sample size'] = N_novrile_anom
                SIE_anom_errors_NO['p-value'] = p_value_anom
                # Append each CV slice to full data set
                SIE_errors_ALL = SIE_errors_ALL.append(SIE_errors)
                SIE_anom_errors_ALL = SIE_anom_errors_ALL.append(SIE_anom_errors)
                SIE_errors_NO_ALL = SIE_errors_NO_ALL.append(SIE_errors_NO)
                SIE_anom_errors_NO_ALL = SIE_anom_errors_NO_ALL.append(SIE_anom_errors_NO)


            # In[14]:


            #SIE_anom_errors_ALL.where(SIE_anom_errors_ALL['region']=='East Siberian-Beaufort-Chukchi Sea').dropna(how='all')


            # Plot RMSE vs lead time!

            # In[15]:


            fig1 = plt.figure(1)
            reg_sel = 'East Siberian-Beaufort-Chukchi Sea'
            ax1 = fig1.add_axes([0,0,1,1])
            foo = SIE_anom_errors_ALL.xs(reg_sel).reset_index()
            foo.plot.scatter(x='lead days',y=['SIE RMSE'],linewidth=2,linestyle='--',ax=ax1,color='r',label=['VRILES'])
            foo2 = SIE_anom_errors_NO_ALL.xs(reg_sel).reset_index()
            foo2.plot.scatter(x='lead days',y=['SIE RMSE'],linewidth=2,linestyle=':',ax=ax1,label=['NO VRILES'])


            # In[16]:


            fig2 = plt.figure(2)
            ax2 = fig2.add_axes([0,0,1,1])
            foo = SIE_anom_errors_ALL.xs(reg_sel).reset_index()
            foo.plot.scatter(x='lead days',y=['SIE anom RMSE'],linewidth=2,linestyle='--',ax=ax2,color='r',label=['VRILES'])
            foo2 = SIE_anom_errors_NO_ALL.xs(reg_sel).reset_index()
            foo2.plot.scatter(x='lead days',y=['SIE anom RMSE'],linewidth=2,linestyle=':',ax=ax2,label=['NO VRILES'])


            # Plot p-values

            # In[17]:


            fig4,(ax4a,ax4b) = plt.subplots(1,2,figsize=(15,6))
            foo = SIE_errors_ALL.xs(reg_sel).reset_index()
            sp4a = sns.scatterplot(data=foo,x='lead days',y='p-value',hue='year out',ax=ax4a,palette='tab20',legend=False)
            ax4a.axhline(-1.96,color='k')
            ax4a.axhline(1.96,color='k')
            SIE_errors_ALL_masked = SIE_errors_ALL.mask(SIE_errors_ALL['p-value'].abs()<1.96)
            foo2 = SIE_errors_ALL_masked.xs(reg_sel).reset_index()
            sns.scatterplot(data=foo2,x='lead days',y='p-value',hue='year out',s=200,legend=False,alpha=0.25,ax=ax4a,palette='tab20')
            sp4a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax4a.set_title('SIE Forecasts')
            #
            sp4b = sns.scatterplot(data=SIE_anom_errors_ALL.xs(reg_sel).reset_index(),x='lead days',
                                   y='p-value',hue='year out',ax=ax4b,palette='tab20')
            ax4b.axhline(-1.96,color='k')
            ax4b.axhline(1.96,color='k')
            SIE_anom_errors_ALL_masked = SIE_anom_errors_ALL.mask(SIE_anom_errors_ALL['p-value'].abs()<1.96)
            sns.scatterplot(data=SIE_anom_errors_ALL_masked.xs(reg_sel).reset_index(),x='lead days',y='p-value',
                            hue='year out',s=200,legend=False,alpha=0.25,ax=ax4b,palette='tab20')
            sp4b.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax4b.set_title('Anomalous SIE Forecasts')
            ax4b.set_xlabel('lead weeks')
            ax4a.set_xlabel('lead weeks')
            #
            fig4.suptitle('p-values, VRILE days vs non-VRILE days, {reg_sel}, {model_name}, {seas_str}'.format(reg_sel=reg_sel,
                                                                        model_name=model_name,seas_str=seas_str),fontsize=20)
            #fpath_save_fig4 = figpath_save+'pvalues_each_fold_{reg_sel}_{model_name}_{seas_str}.png'.format(reg_sel=reg_sel,
            #                                                                    model_name=model_name,seas_str=seas_str)
            #fig4.savefig(fpath_save_fig4,format='png',dpi=350,bbox_inches='tight')

            plt.close()
            # Combine everything together into one data frame. Replace <code>SIE anom RMSE</code> and <code>SIE anom MAE</code> in <code>SIE_errors</code> with corresponding entries from  <code>SIE_anom_errors</code> (and same for raw)

            # In[18]:


            SIE_errors_VRILE = SIE_errors_ALL.copy()
            SIE_errors_VRILE = SIE_errors_VRILE.drop(columns={'SIE anom RMSE','SIE anom MAE'})
            SIE_errors_VRILE = SIE_errors_VRILE.join(SIE_anom_errors_ALL[['SIE anom RMSE','SIE anom MAE']])
            SIE_errors_VRILE['type'] = 'VRILE days'
            #
            SIE_errors_noVRILE = SIE_errors_NO_ALL.copy()
            SIE_errors_noVRILE = SIE_errors_noVRILE.drop(columns={'SIE anom RMSE','SIE anom MAE'})
            SIE_errors_noVRILE = SIE_errors_noVRILE.join(SIE_anom_errors_NO_ALL[['SIE anom RMSE','SIE anom MAE']])
            SIE_errors_noVRILE['type'] = 'no VRILE days'
            #
            SIE_errors_FULL = SIE_errors_VRILE.append(SIE_errors_noVRILE)


            # Save

            # In[19]:


            fdir = '/home/disk/sipn/mcmcgraw/McGraw_etal_2020/code/make_it_nice/COMMON_LAND_MASK/data/{model_name}/'.format(model_name=model_name)
            fdir = fdir+'OBS_{obs_name}/'.format(obs_name=obs_name)
            if COMMON_RF == True:
                fdir = fdir+'COMMON_RF/'
            else:
                fdir = fdir+'FULL_TIME/'
            if nday_change != 5:
                fdir = fdir+'VRILEs_{nday_change}day_change/'.format(nday_change=nday_change)
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            #

            fname_save_RMSE = fdir+'ERROR_PROP_{VRILE_shift}day_shift_VRILE_vs_NOVRILE_RMSE_MAE_{model_name}_months{seas_str}_VRILE{thresh_str}_model_clim_freq_{clim_freq_str}.csv'.format(VRILE_shift=VRILE_shift,
            model_name=model_name,seas_str=seas_str,thresh_str=thresh_str,clim_freq_str=clim_freq_str)
            #
            #SIE_raw_err_FULL.to_csv(fname_save_raw)
            SIE_errors_FULL.to_csv(fname_save_RMSE)


            # In[20]:


            print(fname_save_RMSE)


# In[ ]:




