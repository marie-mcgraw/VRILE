#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:38:15 2019

@author: mcmcgraw
"""

import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from scipy import signal
import scipy.stats as stats

# Marie C. McGraw
# Atmospheric Science, University of Washington
# Updated 07-16-2019
#

filepath = '/home/disk/sipn/nicway/data/model/ecmwfsipn/forecast/sipn_nc_agg/'  #observed sea ice extent data
filename = xr.open_mfdataset(filepath+'/*2018*.nc',concat_dim='init_time')
print(filename)

extent = filename.Extent
#extent = filename.ClimoTrendSIC
print(extent.shape)
#init_time = filename.init_time
#print(init_time)
#region_names = filename.region_names
#print(region_names)
#forecast_time = filename.fore_time
#print(forecast_time)

print(extent.shape)
SIC_mmm = np.nanmean(extent[1,1,:,:,0],axis=0)
SIC_test = np.squeeze(extent[1,1,:,:,0])
#SIC_test = np.nanmean(np.nanmean(extent,axis=2),axis=1)
plt.plot(np.transpose(SIC_test.values))
plt.plot(SIC_mmm,'k')
#plt.plot(SIC_test)
#print(SIC_test.values)

delta_SIC = SIC_test - SIC_mmm
plt.figure()
plt.plot(np.transpose(delta_SIC.values))