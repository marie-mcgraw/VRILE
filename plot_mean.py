#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:12:12 2019

@author: mcmcgraw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# this script calculates and plots basic information about the 5th percentile events of 5-day 
# sea ice extent change (SIE).  The 5th percentile of 5-day change in SIE is our threshold for 
# defining very rapid ice loss events (VRILEs)
"""
Created on Tue Sep  3 15:00:11 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
from datetime import datetime, date
import scipy.stats as stats

##Clear workspace before running 
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

#Load model sea ice extent (SIE) data    
filepath = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
model_name = 'ecmwfsipn'
model_type = 'reforecast'
no_day_change = 5
max_lead = 30

delta_lead_days = np.arange(no_day_change,max_lead+1,1)
delta_first_days = np.arange(1,max_lead+2-no_day_change,1)
no_forecast_periods = len(delta_lead_days)

filepath_load = '/home/disk/sipn/mcmcgraw/data/VRILE/intermediate_data/'
filename_full = filepath_load+'MOVING_{model_name}_{model_type}_d_SIC_{d_days}day_change_lead_time_{lead_days}days_ALL_REGIONS_ALL_ENS.csv'.format(model_name=model_name,
                               model_type=model_type,d_days=no_day_change,
                               lead_days=no_day_change*no_forecast_periods)

file_full = pd.read_csv(filename_full)
regions_list = file_full['region'].unique().tolist()
#plot modeled data
