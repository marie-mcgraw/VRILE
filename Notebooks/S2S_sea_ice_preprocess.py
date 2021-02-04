import numpy as np
import pandas as pd
import xarray as xr
import os

#1. Load S2S model output from Nic's files. We want the control runs too.  Then, select only the common reforecast period (1999-01-01 to 2014-12-31), and add the control run to the rest of the output. NOTE: for NCEP, since the reforecast period is short (ends in 2011), we will just use the entire period. 

def load_model(model_name):
    # Paths for perturb and control runs
    filepath = '/home/disk/sipn/nicway/data/model/{model_name}/reforecast/sipn_nc_agg/'.format(model_name=model_name)
    filepath_ctrl = '/home/disk/sipn/nicway/data/model/{model_name}/reforecast.control/sipn_nc_agg/'.format(model_name=model_name)
    # Open both with xarray
    filenames = xr.open_mfdataset(filepath+'/*.nc',combine='by_coords')
    filenames_ctrl = xr.open_mfdataset(filepath_ctrl+'/*.nc',combine='by_coords')
    print(filenames)
    # load SIE
    SIE = filenames.Extent
    SIE_ctrl = filenames_ctrl.Extent
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
        SIE = SIE.sel(init_time=slice(common_start,common_end))
        SIE_ctrl = SIE_ctrl.sel(init_time=slice(common_start,common_end))
        # Remove repeated indices in CTRL
        _,init_ind_c = np.unique(SIE_ctrl['init_time'],return_index=True)
        SIE_ctrl = SIE_ctrl.isel(init_time=init_ind_c)
    # Concatenate the two
    SIE = xr.concat([SIE,SIE_ctrl],dim='ensemble')
    
    return SIE

#2.  Create aggregate regions from NSIDC MASIE regions to simplify our data a bit. We want to create a few aggregate regions from the NSIDC MASIE regions (more at: https://nsidc.org/data/masie/browse_regions) since some regions are so small. We're going to combine the following:
    #Kara and Laptev Seas (region_KL)
    #Barents, Kara and Laptev Seas (region_BKL)
    #East Siberian, Beaufort, and Chukchi Seas (region_EBC)
    #Atlantic (Baffin Bay and East Greenland Sea) (region_ATL)
    #East Siberian, Beaufort, Chukchi, Laptev Seas (region_EBCL)
def create_aggregate_regions(SIE_data):
    nregions = SIE_data['nregions']
    region_names = SIE_data['region_names']
    # Get corresponding indices for each of our aggregate regions
    region_KL = nregions[region_names.isin(['Kara Sea','Laptev Sea'])]
    region_BKL = nregions[region_names.isin(['Barents Sea','Kara Sea','Laptev Sea'])]
    region_EBC = nregions[region_names.isin(['East Siberian Sea','Beaufort Sea','Chukchi Sea'])]
    region_ATL = nregions[region_names.isin(['Baffin Bay','East Greenland Sea'])]
    region_EBCL = nregions[region_names.isin(['East Siberian Sea','Beaufort Sea','Chukchi Sea','Laptev Sea'])]
    # Select each aggregate region, add them together, and add the 'nregions' dimension back; concatenate all aggregates 
    SIE_agg = xr.concat([SIE_data.sel(nregions=region_KL).sum(dim='nregions').expand_dims(dim='nregions'),
                  SIE_data.sel(nregions=region_BKL).sum(dim='nregions').expand_dims(dim='nregions'),
                  SIE_data.sel(nregions=region_EBC).sum(dim='nregions').expand_dims(dim='nregions'),
                  SIE_data.sel(nregions=region_ATL).sum(dim='nregions').expand_dims(dim='nregions'),
                  SIE_data.sel(nregions=region_EBCL).sum(dim='nregions').expand_dims(dim='nregions')],dim='nregions')
    # Add coordinates to nregions.  Start at 20 to make a clear separation from original NSIDC regions
    SIE_agg = SIE_agg.assign_coords(nregions=[20,21,22,23,24])
    # Add region names
    region_names_extra = ['Kara-Laptev Sea','Barents-Kara-Laptev Sea','East Siberian-Beaufort-Chukchi Sea',
                      'Atlantic','East Siberian-Beaufort-Chukchi-Laptev Sea']
    SIE_agg["region_names"] = ("nregions",region_names_extra)
    #SIE
    SIE_data = xr.concat([SIE_data,SIE_agg],dim='nregions')
    return(SIE_data)

#3. Create climatology for model output based on region, day of year, and lead time. Get month-day for valid dates (don't use dayofyear because of leap days). Since our forecasts are not initialized every day, we will do two versions--one where we keep each lead time separate, and one where we group our lead time climatology based on week instead of day (which is more supported in the literature).

def create_model_climatology(SIE,week_length):
    # Add valid date in %m-%d format
    SIE['valid date of yr'] = pd.to_datetime(SIE['valid date']).dt.strftime('%m-%d')
    # Determine lead time as a function of weeks instead of days
    SIE_df_weekly = SIE.copy()
    SIE_df_weekly['lead time (weeks)'] = SIE_df_weekly['lead time (days)'].values.astype('timedelta64[D]')/pd.Timedelta(week_length,'D')
    SIE_df_weekly['lead time (weeks)'] = SIE_df_weekly['lead time (weeks)'].apply(np.floor)
    # Group by region, lead time, and valid day of year. Use .transform('mean') so that our climatology has the 
    # same shape as the original input dataframe (so we can just subtract SIE_clim from SIE easily at the end)
    SIE['SIE clim'] = SIE.groupby(['region','lead time (days)','valid date of yr'])['SIE'].transform('mean')
    SIE_df_weekly['SIE clim'] = SIE_df_weekly.groupby(['region','lead time (weeks)','valid date of yr'])['SIE'].transform('mean')

    SIE['SIE anom'] = SIE['SIE'] - SIE['SIE clim']
    SIE_df_weekly['SIE anom'] = SIE_df_weekly['SIE'] - SIE_df_weekly['SIE clim']
    return SIE
#3b.  Model climatology using rolling 
#4. Climatology for observations (this is easier since we don't have to think about lead time)
def create_obs_climatology(SIE):
    # Add valid date in %m-%d format
    SIE['valid day of year'] = pd.to_datetime(SIE['valid date']).dt.strftime('%m-%d')
    # Group by region and day of year and take the mean. Use transform to make the output match the dataframe instead
    # of creating a multi-index
    SIE['SIE clim'] = SIE.groupby(['region','valid day of year'])['SIE'].transform('mean')
    # And simply subtract SIE_clim from actual SIE
    SIE['SIE anom'] = SIE['SIE'] - SIE['SIE clim']
    return SIE


#5.  Rolling mean climatology for observations--an alternative way to calculate the climatology by using an <code>n_years</code> year rolling mean.  The advantage of this approach is that it can somewhat account for the trends in sea ice by "moving the box" a little bit in time.  
def create_obs_climatology_ROLLING(SIE,n_years):
    # Add valid date in %m-%d format
    SIE['valid day of year'] = pd.to_datetime(SIE['valid date']).dt.strftime('%m-%d')
    # Use past n_years
    
    # Group by region and day of year and take the mean. Use transform to make the output match the dataframe instead
    # of creating a multi-index. Rolling will take the past n_years years for each day of the year
    SIE['SIE clim'] = SIE.groupby(['region','valid day of year'])['SIE'].transform(lambda x: x.rolling(n_years,min_periods=n_years).mean())
    # And simply subtract SIE_clim from actual SIE
    SIE['SIE anom'] = SIE['SIE'] - SIE['SIE clim']
    return SIE
