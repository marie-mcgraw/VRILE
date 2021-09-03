#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
from S2S_sea_ice_preprocess import create_obs_climatology
from S2S_sea_ice_VRILEs import get_VRILE_days


# In[2]:


def load_model_SIC_by_year(model_name,year):
    # Paths for perturb and control runs
    filepath = '/home/disk/sipn/nicway/data/model/{model_name}/reforecast/sipn_nc/'.format(model_name=model_name)
    filepath_ctrl = '/home/disk/sipn/nicway/data/model/{model_name}/reforecast.control/sipn_nc/'.format(model_name=model_name)
    # Open both with xarray
    filenames = xr.open_mfdataset(filepath+'/{model_name}_{year}_*.nc'.format(model_name=model_name,year=year)
                                  ,combine='by_coords')#,
                                  #chunks={'ensemble': 6, 'init_time': 1, 'fore_time': 60,'y':102,'x':112})
    filenames_ctrl = xr.open_mfdataset(filepath_ctrl+'/{model_name}_{year}_*.nc'.format(model_name=model_name,year=year),
                                       combine='by_coords')#,
                                  #chunks={'lon':102,'lon':112})
    print(filenames)
    # load SIE
    SIE = filenames.sic
    # Rename ni to y, nj to x
    SIE = SIE.rename({'ni':'y','nj':'x'})
    SIE_ctrl = filenames_ctrl.sic
    SIE_ctrl = SIE_ctrl.rename({'ni':'y','nj':'x'})
    # Add coordinate to ensemble dimension for SIE_ctrl so we can combine with SIE
    SIE_ctrl.coords['ensemble'] = xr.DataArray([len(SIE.ensemble)],
                                               dims='ensemble', coords={'ensemble':[len(SIE.ensemble)]})
    # Use common reforecast period for all models EXCEPT NCEP
    if model_name == 'ncep':
        # Don't need to use a common reforecast but remove repeated indices
        _,init_ind = np.unique(SIE['init_time'],return_index=True)
        _,init_ind_c = np.unique(SIE_ctrl['init_time'],return_index=True)
        SIE = SIE.isel(init_time=init_ind)
        SIE_ctrl = SIE_ctrl.isel(init_time=init_ind_c)
    elif model_name != 'ncep':
        common_start = '1999-01-01'
        common_end = '2014-12-31'
        # Select only common reforecast period (full period for NCEP)
        #SIE = SIE.sel(init_time=slice(common_start,common_end))
        #SIE_ctrl = SIE_ctrl.sel(init_time=slice(common_start,common_end))
        # Remove repeated indices in CTRL
        _,init_ind_c = np.unique(SIE_ctrl['init_time'],return_index=True)
        SIE_ctrl = SIE_ctrl.isel(init_time=init_ind_c)
    # Concatenate the two
    SIE = xr.concat([SIE,SIE_ctrl],dim='ensemble')
    
    return SIE


# In[3]:


model_name = 'ukmo'
region_sel = 'Kara Sea'
nregion = 9
years_all = np.arange(1999,2015)
for year_sel in years_all:
    print('running year ',year_sel,' for ',model_name,' model')
    SIC_load = load_model_SIC_by_year(model_name,year_sel)
    SIC_load = SIC_load.mean(dim='ensemble')


    # In[4]:


    obs_load_path = '/home/disk/sipn/nicway/data/obs/NSIDC_0079/sipn_nc_yearly/'
    obs_load = xr.open_mfdataset(obs_load_path+'*.nc',concat_dim='time')
    obs_load = obs_load
    # Trim to 1999-2014
    obs_load = obs_load.where((obs_load['time.year']==year_sel),drop=True)


    # In[5]:


    ds_region = xr.open_dataset('/home/disk/sipn/nicway/data/grids/sio_2016_mask_Update.nc')


    # Load observed SIE to define VRILE days

    # In[6]:


    obs_type = 'sipn_nc_yearly_agg'
    obs_name = 'NSIDC_0079'
    filepath = '/home/disk/sipn/nicway/data/obs/{model_name}/{model_type}/'.format(model_name=obs_name,
                                                                                  model_type=obs_type)
    obs_filenames = xr.open_mfdataset(filepath+'/*.nc',combine='by_coords')
    print('opening ',obs_filenames)
    obs_SIEx = obs_filenames.Extent
    obs_regions = obs_filenames.nregions
    obs_region_names = obs_filenames['region_names'].values
    # Drop region names and re-add as a non-dask.array object.  This is stupid but oh well
    obs_SIEx = obs_SIEx.drop('region_names')
    obs_SIEx["region_names"] = ("nregions",obs_region_names)
    print('obs loaded')
    obs_SIE = obs_SIEx.to_dataframe().reset_index()
    obs_SIE = obs_SIE.rename(columns={'Extent':'SIE','region_names':'region','time':'valid date'})


    # Climatology for observed SIE

    # In[7]:


    obs_SIE = obs_SIE[pd.to_datetime(obs_SIE['valid date']).dt.year.isin(np.arange(1999,2015))]
    obs_SIE = create_obs_climatology(obs_SIE)


    # Get VRILE days

    # In[8]:


    vrile_thresh = 0.05
    nday_change = 5
    seas_sel = [6,7,8]
    seas_str = 'JJA'
    obs_SIE_VRILE_onlyx, obs_SIE_anom_VRILE_onlyx, obs_SIE_NO_VRILEx, obs_SIE_anom_NO_VRILEx = get_VRILE_days(obs_SIE,vrile_thresh,nday_change,seas_sel)
    print('VRILE days calculated')

    #obs_SIEx = obs_SIEx.where(obs_SIEx.region_names==region_sel,drop=True)


    # In[9]:



    VRILE_dates = obs_SIE_anom_VRILE_onlyx[['region','valid date']]
    VRILE_dates_sel = VRILE_dates.where(VRILE_dates['region']==region_sel).dropna(how='all')
    noVRILE_dates = obs_SIE_anom_NO_VRILEx[['region','valid date']]
    noVRILE_dates_sel = noVRILE_dates.where(noVRILE_dates['region']==region_sel).dropna(how='all')
    mon_sel = [6,7,8]
    VRILE_dates_sel_mon = VRILE_dates_sel[pd.to_datetime(VRILE_dates_sel['valid date']).dt.month.isin(mon_sel)]
    noVRILE_dates_sel_mon = noVRILE_dates_sel[pd.to_datetime(noVRILE_dates_sel['valid date']).dt.month.isin(mon_sel)]


    # In[10]:


    obs_SIE_VRILE = obs_load.where(obs_load.time.isin(VRILE_dates_sel['valid date']),drop=True)
    obs_SIE_noVRILE = obs_load.where(obs_load.time.isin(noVRILE_dates_sel['valid date']),drop=True)


    # In[11]:

    ds_region_x = ds_region.sel(nregions=nregion,ocean_regions=nregion)


    # Get IIEE

    # In[ ]:


    sic_thresh = 0.15
    for i_fore in np.arange(0,len(SIC_load.fore_time)):
        print('forecast lead time is ',i_fore)
        SIC_sel = SIC_load.isel(fore_time=i_fore)#.mean(dim='init_doy')
        v_time = SIC_sel['init_time'] + pd.Timedelta(i_fore,'D')
        VRILE_date_shifted = VRILE_dates_sel_mon['valid date'] - pd.Timedelta(i_fore,'D')
        SIC_sel = SIC_sel.assign_coords(valid_date = ("init_time",v_time))

        test_VRILE = SIC_sel.where(SIC_sel.init_time.isin(VRILE_date_shifted),drop=True)
        test_noVRILE = SIC_sel.where(~SIC_sel.init_time.isin(VRILE_date_shifted),drop=True)

        test_noVRILE_trim = test_noVRILE.where(test_noVRILE.valid_date.dt.month.isin([6,7,8]),drop=True)

        SIC_VRILE_sip = (test_VRILE >= sic_thresh).where(test_VRILE.notnull())
        SIC_nonVRILE_sip = (test_noVRILE_trim >= sic_thresh).where(test_noVRILE_trim.notnull())

        SIC_VRILE_sip = SIC_VRILE_sip.where(ds_region_x.mask==nregion)
        #SIC_nonVRILE_sip = SIC_nonVRILE_sip.where(ds_region.mask.isin(ds_region.ocean_regions))
        SIC_nonVRILE_sip = SIC_nonVRILE_sip.where(ds_region_x.mask==nregion)

        # Same for obs (and rename dimension to match model)
        obs_VRILE_sip = (obs_SIE_VRILE >= sic_thresh).where(obs_SIE_VRILE.notnull())#.mean(dim='init_doy')
        #obs_VRILE_sip = obs_VRILE_sip.where(ds_region.mask.isin(ds_region.ocean_regions))
        obs_VRILE_sip = obs_VRILE_sip.where(ds_region_x.mask==nregion)
        obs_noVRILE_sip = (obs_SIE_noVRILE >= sic_thresh).where(obs_SIE_noVRILE.notnull())#.mean(dim='init_doy')
        obs_noVRILE_sip = obs_noVRILE_sip.where(ds_region_x.mask==nregion)

        #obs_VRILE_sip = obs_VRILE_sip.assign_coords(time = ("init_time",SIC_VRILE_sip.init_time))
        obs_VRILE_sip = obs_VRILE_sip.rename({'time':'valid_date'})
        obs_noVRILE_sip = obs_noVRILE_sip.rename({'time':'valid_date'})
        #
        # Finally, keep only obs days that are in the model data set
        valid_keep_VRILE = SIC_VRILE_sip.init_time + pd.Timedelta(i_fore,'D')
        valid_keep_noVRILE = SIC_nonVRILE_sip.init_time + pd.Timedelta(i_fore,'D')
        #obs_VRILE_sip_trim = obs_VRILE_sip.where(obs_VRILE_sip.init_time.isin(valid_keep_VRILE),drop=True)
        #obs_noVRILE_sip_trim = obs_noVRILE_sip.where(obs_noVRILE_sip.init_time.isin(valid_keep_noVRILE))
        obs_VRILE_sip_trim = obs_VRILE_sip.where(obs_VRILE_sip.valid_date.isin(valid_keep_VRILE),drop=True)
        obs_noVRILE_sip_trim = obs_noVRILE_sip.where(obs_noVRILE_sip.valid_date.isin(valid_keep_noVRILE),drop=True)
        # get IIEE

        obs_VRILE_sip_trim = obs_VRILE_sip.where(obs_VRILE_sip.valid_date.isin(valid_keep_VRILE),drop=True)
        obs_VRILE_sip_trim_v = obs_VRILE_sip_trim.assign_coords(valid_date = ("init_time",SIC_VRILE_sip.init_time))
        # Get IIEE
        abs_diff_VRILE = (abs(SIC_VRILE_sip - obs_VRILE_sip_trim_v['sic'])*ds_region_x.area).sum(dim=['x','y'])/(10**6)
        abs_diff_noVRILE = (abs(SIC_nonVRILE_sip - obs_noVRILE_sip_trim['sic'])*ds_region_x.area).sum(dim=['x','y'])/(10**6)
        abs_diff_VRILE = abs_diff_VRILE.expand_dims({'fore_time':1})
        abs_diff_noVRILE = abs_diff_noVRILE.expand_dims({'fore_time':1})
        if i_fore == 0:
            IIEE_VRILE = abs_diff_VRILE.mean(dim=['init_time','valid_date'])
            IIEE_noVRILE = abs_diff_noVRILE.mean(dim=['init_time','valid_date'])

        else:
            IIEE_VRILE = xr.concat([IIEE_VRILE,abs_diff_VRILE.mean(dim=['init_time','valid_date'])],dim='fore_time')
            IIEE_noVRILE = xr.concat([IIEE_noVRILE,abs_diff_noVRILE.mean(dim=['init_time','valid_date'])],dim='fore_time')
        #IIEE_VRILE = xr.concat([IIEE_VRILE,abs_diff_VRILE.mean(dim=['init_time','valid_date'])],dim='fore_time')
        #IIEE_noVRILE = xr.concat([IIEE_noVRILE,abs_diff_noVRILE.mean(dim=['init_time','valid_date'])],dim='fore_time')


    # Convert to DataFrame

    # In[ ]:


    IIEE_df = IIEE_VRILE.to_dataframe(name='IIEE')
    IIEE_df_no = IIEE_noVRILE.to_dataframe(name='IIEE')
    IIEE_df['type'] = 'VRILE days'
    IIEE_df_no['type'] = 'no VRILE days'
    IIEE_df_all = IIEE_df.append(IIEE_df_no)
    IIEE_df_all['year'] = year_sel
    IIEE_df_all['season'] = seas_str
    IIEE_df_all['region_names'] = region_sel


    # If a file for this model doesn't exist, create it.  Otherwise, append to the existing file

    # In[ ]:


    fdir =  '../../data/IIEE/'
    fname_save = fdir+'IIEE_all_{model_name}_{seas_str}.csv'.format(model_name=model_name,seas_str=seas_str)
    # Make directory
    if not os.path.exists(fdir):
        os.makedirs(fdir) 
        print('creating new directory')
    # 
    if not os.path.exists(fname_save):
        IIEE_df_all.to_csv(fname_save)
        print('creating new file')
    else:
        with open(fname_save, 'a') as f:
            IIEE_df_all.to_csv(f, header=False)
            print('adding to file')


    # In[ ]:


  #  IIEE_df_all.head(30)


    # In[ ]:




