#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:59:33 2019

@author: mcmcgraw
"""

import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import scipy.stats as stats
from netCDF4 import Dataset
 
file_path_Z500 = '/home/disk/sipn/rclancy/ecmwf/cf/native/Z500'
#filename = xr.open_mfdataset(file_path_Z500+'/*.netcdf',concat_dim='time')
fart = Dataset(file_path_Z500+'/ecmwf_2004-07-16.netcdf','r')
Z_test = fart.variables['gh']
time = fart.variables['time']
time2 = np.ma.getdata(time).data
start_date = datetime(1900,1,1)
d2 = timedelta(hours=time2[6])
delt = d2 - start_date