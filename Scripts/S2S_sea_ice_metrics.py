import numpy as np
import pandas as pd
import xarray as xr
import os

# Calculate raw error, mean absolute error, and RMSE.  These calculations will be a function of region and lead time.  See nice notebook for equations for RMSE and MAE
def calculate_errors(model_SIE,obs_SIE):
    # Group by region, valid date, and lead time (for model)
    SIE_obsx = obs_SIE.groupby(['region','valid date'])['SIE','SIE clim','SIE anom'].mean()
    # First calculate raw model error (we'll also group by init time so we can save that for calculating other stuff)
    SIE_model_raw = model_SIE.groupby(['region','valid date','lead days','init date'])['SIE','SIE clim','SIE anom'].mean()
    SIE_raw_error = SIE_model_raw[['SIE','SIE anom']] - SIE_obsx[['SIE','SIE anom']]
    SIE_raw_error = SIE_raw_error.dropna(how='all')
    # Now, we'll do MAE and RMSE (we don't care about the date of initialization)
    SIE_modelx = model_SIE.groupby(['region','valid date','lead days'])['SIE','SIE clim','SIE anom'].mean()
    SIE_diff = SIE_modelx[['SIE','SIE anom']] - SIE_obsx[['SIE','SIE anom']]
    # Square errors to get RMSE. get absolute value of errors for MAE
    SIE_diff = SIE_diff.dropna(how='all')
    SIE_diff = SIE_diff.rename(columns={'SIE':'SIE raw error','SIE anom':'SIE anom raw error'})
    SIE_diff[['SIE sq error','SIE anom sq error']] = SIE_diff**2
    # Now, average over all valid dates and take the square root. 
    SIE_errors = SIE_diff[['SIE sq error','SIE anom sq error']].mean(level=(0,2))**0.5
    # Absolute value, then average over all valid dates to get the MAE
    SIE_errors[['SIE MAE','SIE anom MAE']] = SIE_diff[['SIE raw error','SIE anom raw error']].abs().mean(level=(0,2))
    SIE_errors = SIE_errors.rename(columns={'SIE sq error':'SIE RMSE','SIE anom sq error':'SIE anom RMSE'})
    return(SIE_raw_error,SIE_errors)
    
# Get p-value to determine whether or not we can reject the null hypothesis that forecast error on VRILE days is not different from forecast error on non-VRILE days    
def get_pvalues(SIE_VRILE,SIE_noVRILE,SIE_error_VRILE,SIE_error_noVRILE):
    # Standard deviation
    sdev_vrile = SIE_VRILE.groupby(['region','lead days'])['SIE'].std()
    sdev_novrile = SIE_noVRILE.groupby(['region','lead days'])['SIE'].std()
    # p-value
    pval_num = (SIE_error_VRILE['SIE RMSE'] - SIE_error_noVRILE['SIE RMSE'])
    N_vrile = len(SIE_VRILE)
    N_novrile = len(SIE_noVRILE)
    pval_den = np.sqrt(((sdev_vrile**2)/N_vrile)+((sdev_novrile**2)/N_novrile))
    pval = pval_num/pval_den
    # 
    pval = pval.rename('p-value')
    return sdev_vrile,sdev_novrile,pval,N_vrile,N_novrile

# Create a damped anomaly forecast
def calc_damped_anomaly(obs,alpha,max_fore,days_of_yr):
    SIE_forecast = pd.DataFrame(columns=['init date','valid date','lead time','SIE','SIE anom','SIE clim','region'])
    for i_doy in np.arange(1,366):
        #print('running day of year: ',i_doy)
        for i_fore in np.arange(1,max_fore+1):
            #i_fore = 45
            if i_doy + i_fore > 365:
                f_doy = (i_doy + i_fore) - 365
            else:
                f_doy = i_doy + i_fore
            # obs for the days we are forecasting
            obs_sel = obs.where(obs['valid day of year'].isin(days_of_yr.loc[i_doy].reset_index(drop=True).values)).dropna(how='all')
            # 
            d1 = obs_sel['SIE anom']*alpha.sel(init_time=i_doy,fore_time=i_fore).values
            obs_clim_sel = obs.where(obs['valid day of year'].isin(days_of_yr.loc[f_doy].reset_index(drop=True).values)).dropna(how='all')
            d2 = d1.values+obs_clim_sel['SIE clim'].values
            #obs_sel.plot(y=['SIE','SIE clim'])
            d3 = pd.DataFrame()
            d3['init date'] = obs_sel['valid date']
            d3['valid date'] = obs_sel['valid date'] + pd.Timedelta(i_fore,'D')
            d3['lead time'] = pd.Timedelta(i_fore,'D')
            d3['SIE'] = d2
            d3['SIE anom'] = d1.values
            d3['SIE clim'] = obs_sel['SIE clim']
            d3['region'] = obs_sel['region']
            #
            SIE_forecast = SIE_forecast.append(d3)
    #
    return SIE_forecast