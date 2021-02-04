import numpy as np
import pandas as pd
import xarray as xr
import os

# Functions for calculating forecast model errors

# [1] Calculate raw error, mean absolute error, and RMSE. These calculations will be a function of region and lead time. Equations for RMSE and MAE are in Jupyter notebook
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