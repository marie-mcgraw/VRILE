import numpy as np
import pandas as pd
import xarray as xr
import os

def get_VRILE_days(obs_SIE,vrile_thresh,nday_change,seas_sel):
    # Use shift to move SIE/SIE anom foreward and backward by nday_change days
    nday_shift = np.floor(nday_change/2)
    obs_SIE['SIE anom -n'] = obs_SIE.groupby(['region'])['SIE anom'].shift(-nday_shift)
    obs_SIE['SIE anom +n'] = obs_SIE.groupby(['region'])['SIE anom'].shift(+nday_shift)
    obs_SIE['SIE -n'] = obs_SIE.groupby(['region'])['SIE'].shift(-nday_shift)
    obs_SIE['SIE +n'] = obs_SIE.groupby(['region'])['SIE'].shift(+nday_shift)
    obs_SIE['d_SIE'] = obs_SIE['SIE -n'] - obs_SIE['SIE +n']
    obs_SIE['d_SIE anom'] = obs_SIE['SIE anom -n'] - obs_SIE['SIE anom +n']
    # Trim to correct season based on seas_sel
    if not seas_sel:
        obs_SIE = obs_SIE
    else:
        obs_SIE = obs_SIE[pd.to_datetime(obs_SIE['valid date']).dt.month.isin(seas_sel)]
    # Identify VRILE days based on vrile_thresh
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
    obs_SIE_NO_VRILE = obs_SIE_NO_VRILE.reset_index(level=0,drop=True).reset_index()
    obs_SIE_anom_NO_VRILE = obs_SIE_anom_NO_VRILE.reset_index(level=0,drop=True).reset_index()
    return obs_SIE_VRILE_only, obs_SIE_anom_VRILE_only, obs_SIE_NO_VRILE, obs_SIE_anom_NO_VRILE

# Get VRILE event days--track consecutive days and make sure events are separated by at least BUFFER_DAYS
def get_VRILE_days_EVENTS(obs_SIE,vrile_thresh,nday_change,seas_sel,buffer_days):
    # Use shift to move SIE/SIE anom foreward and backward by nday_change days
    nday_shift = np.floor(nday_change/2)
    obs_SIE['SIE anom -n'] = obs_SIE.groupby(['region'])['SIE anom'].shift(-nday_shift)
    obs_SIE['SIE anom +n'] = obs_SIE.groupby(['region'])['SIE anom'].shift(+nday_shift)
    obs_SIE['SIE -n'] = obs_SIE.groupby(['region'])['SIE'].shift(-nday_shift)
    obs_SIE['SIE +n'] = obs_SIE.groupby(['region'])['SIE'].shift(+nday_shift)
    obs_SIE['d_SIE'] = obs_SIE['SIE -n'] - obs_SIE['SIE +n']
    obs_SIE['d_SIE anom'] = obs_SIE['SIE anom -n'] - obs_SIE['SIE anom +n']
    # Trim to correct season based on seas_sel
    if not seas_sel:
        obs_SIE = obs_SIE
    else:
        obs_SIE = obs_SIE[pd.to_datetime(obs_SIE['valid date']).dt.month.isin(seas_sel)]
    # Identify VRILE days based on vrile_thresh
    obs_SIE['SIE p05'] = obs_SIE.groupby(['region'])['d_SIE'].transform(lambda x: x.quantile(vrile_thresh))
    obs_SIE['SIE anom p05'] = obs_SIE.groupby(['region'])['d_SIE anom'].transform(lambda x: x.quantile(vrile_thresh))
    obs_SIE_VRILE_only = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE'] <= x['SIE p05']])
    obs_SIE_anom_VRILE_only = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE anom'] <= x['SIE anom p05']])
    # Now, identify out EVENTS--we only want VRILE days that happen after at least BUFFER_DAYS of no VRILEs
    # Our very first VRILE day will be NaT; mask that with 500 
    obs_SIE_VRILE_only = obs_SIE_VRILE_only.drop(columns='region').reset_index()
    obs_SIE_anom_VRILE_only = obs_SIE_anom_VRILE_only.drop(columns='region').reset_index()
    shift_dates = lambda x: x - x.shift(1)
    obs_SIE_VRILE_only['valid date diff'] = obs_SIE_VRILE_only.groupby(['region'])['valid date'].transform(shift_dates)
    obs_SIE_VRILE_only['valid date diff'] = obs_SIE_VRILE_only['valid date diff'].fillna(pd.Timedelta(500,'D'))
    #
    obs_SIE_anom_VRILE_only['valid date diff'] = obs_SIE_anom_VRILE_only.groupby(['region'])['valid date'].transform(shift_dates)
    obs_SIE_anom_VRILE_only['valid date diff'] = obs_SIE_anom_VRILE_only['valid date diff'].fillna(pd.Timedelta(500,'D'))
    # 
    obs_SIE_VRILE_only = obs_SIE_VRILE_only.where(obs_SIE_VRILE_only['valid date diff']>= pd.Timedelta(buffer_days,'D')).dropna(how='all')
    obs_SIE_anom_VRILE_only = obs_SIE_anom_VRILE_only.where(obs_SIE_anom_VRILE_only['valid date diff']>= pd.Timedelta(buffer_days,'D')).dropna(how='all')
    # Also return non-VRILE days only
    obs_SIE_NO_VRILE = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE'] > x['SIE p05']])
    obs_SIE_anom_NO_VRILE = obs_SIE.groupby(['region']).apply(lambda x: x[x['d_SIE anom'] > x['SIE anom p05']])
    # Keep only VRILE days
    obs_SIE_VRILE_only = obs_SIE_VRILE_only.reset_index(level=0, drop=True).reset_index()
    obs_SIE_anom_VRILE_only = obs_SIE_anom_VRILE_only.reset_index(level=0, drop=True).reset_index()
    obs_SIE_NO_VRILE = obs_SIE_NO_VRILE.reset_index(level=0,drop=True).reset_index()
    obs_SIE_anom_NO_VRILE = obs_SIE_anom_NO_VRILE.reset_index(level=0,drop=True).reset_index()
    return obs_SIE_VRILE_only, obs_SIE_anom_VRILE_only, obs_SIE_NO_VRILE, obs_SIE_anom_NO_VRILE