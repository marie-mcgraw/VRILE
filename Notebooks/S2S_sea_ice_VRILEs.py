import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import glob2
from scipy import signal 
import seaborn as sns
import os

# [1] Calculate VRILE days from obs.  Determine $n$-day changes in SIE and anomalous SIE, and estimate the lower tail of the distribution based on <code>VRILE_thresh</code>.  

def get_VRILE_days(obs_SIE,vrile_thresh,nday_change,seas_sel,seas_str):
    # Use shift to move SIE/SIE anom foreward and backward by nday_change days
    nday_shift = np.floor(nday_change/2)
    obs_SIE['SIE anom -n'] = obs_SIE.groupby(['region'])['SIE anom'].shift(-nday_shift)
    obs_SIE['SIE anom +n'] = obs_SIE.groupby(['region'])['SIE anom'].shift(+nday_shift)
    obs_SIE['SIE -n'] = obs_SIE.groupby(['region'])['SIE'].shift(-nday_shift)
    obs_SIE['SIE +n'] = obs_SIE.groupby(['region'])['SIE'].shift(+nday_shift)
    obs_SIE['d_SIE'] = obs_SIE['SIE -n'] - obs_SIE['SIE +n']
    obs_SIE['d_SIE anom'] = obs_SIE['SIE anom -n'] - obs_SIE['SIE anom +n']
    # Trim to correct season based on seas_sel
    if seas_str == 'ALL':
        obs_SIE['SIE p05'] = obs_SIE.groupby(['region'])['d_SIE'].transform(lambda x: x.quantile(vrile_thresh))
        obs_SIE['SIE anom p05'] = obs_SIE.groupby(['region'])['d_SIE anom'].transform(lambda x: x.quantile(vrile_thresh))
    # If we are looking at VRILEs in only one season, trim to those months only
    else:
        obs_SIE = obs_SIE[pd.to_datetime(obs_SIE['valid date']).dt.month.isin(seas_sel)]
        obs_SIE['SIE p05'] = obs_SIE.groupby(['region'])['d_SIE'].transform(lambda x: x.quantile(vrile_thresh))
        obs_SIE['SIE anom p05'] = obs_SIE.groupby(['region'])['d_SIE anom'].transform(lambda x: x.quantile(vrile_thresh))
    obs_SIE_VRILE_only = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE'] <= x['SIE p05']])
    obs_SIE_anom_VRILE_only = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE anom'] <= x['SIE anom p05']])
    # Also return non-VRILE days only
    obs_SIE_NO_VRILE = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE'] > x['SIE p05']])
    obs_SIE_anom_NO_VRILE = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE anom'] > x['SIE anom p05']])
    # Keep only VRILE days
    obs_SIE_VRILE_only = obs_SIE_VRILE_only.reset_index(level=0, drop=True).reset_index()
    obs_SIE_anom_VRILE_only = obs_SIE_anom_VRILE_only.reset_index(level=0, drop=True).reset_index()
    #
    obs_SIE_NO_VRILE = obs_SIE_NO_VRILE.reset_index(level=0,drop=True).reset_index()
    obs_SIE_anom_NO_VRILE = obs_SIE_anom_NO_VRILE.reset_index(level=0,drop=True).reset_index()
    
    return obs_SIE_VRILE_only, obs_SIE_anom_VRILE_only, obs_SIE_NO_VRILE, obs_SIE_anom_NO_VRILE
