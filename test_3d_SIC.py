#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:53:44 2019

@author: mcmcgraw
"""

import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import datetime
import calendar
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import xarray as xr

#filepath = '/home/disk/sipn/nicway/data/obs/NSIDC_0079/sipn_nc_yearly/'  #observed sea ice extent data
filepath = '/home/disk/sipn/nicway/data/obs/NSIDC_0079/sipn_nc_yearly_agg/'  #observed sea ice extent data
file_list = xr.open_mfdataset(filepath+'/*.nc',concat_dim='time')
#
region_names = file_list.region_names
#print(region_names)
print(file_list)
extent_regions = file_list.Extent
extent = extent_regions.array
time_load = file_list.time
print(extent_regions)
#print(time)
#plt.plot(extent_regions[:,0])
#SIC = file_list.sic
#time_load = file_list.time
#print(SIC.shape)
TIME = np.transpose(np.stack([np.array(time_load.dt.year),np.array(time_load.dt.month),np.array(time_load.dt.day)]))
#print(TIME)
#lat = file_list.lat
#lon = file_list.lon


#f_SIC = netCDF4.Dataset(filepath+filename)
#
##show variables
#print(f_SIC.variables.keys())
##
#SIC = f_SIC.variables['sic']
#extent = f_SIC.variables['extent']
#print(SIC.shape)
##SIC = np.transpose(SIC)
##print(SIC.shape)
##
##lat = f_SIC.variables['lat']
##lon = f_SIC.variables['lon']
##
#time = f_SIC.variables['time']
#print(time[0:10])
##fig1 = plt.figure()
##plt.contourf(SIC,20)
##plt.colorbar