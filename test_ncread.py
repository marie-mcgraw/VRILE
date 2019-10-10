# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import datetime
import calendar
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns

filepath = '/home/disk/sipn/nicway/data/obs/NSIDC_extent/sipn_nc/'  #observed sea ice extent data
filename = 'N_seaice_extent_daily_v3.0.nc'

f_SIC = netCDF4.Dataset(filepath+filename)

#show variables
print(f_SIC.variables.keys())

SIC = f_SIC.variables['Extent']
#time variable. We want to 
time = f_SIC.variables['datetime'] #days since 10-26-1978
days_since = np.array([])
dates_since = np.array([])
no_yrs = len(SIC)
start_date = datetime(1978,10,26,0,0)
years = np.array([])
months = np.array([])
days = np.array([])

for iday in np.arange(0,len(SIC)):
    #days since 10-26-1978
    day_i = time[iday]
    days_since = np.append(days_since,day_i.data)
    #convert to date
    delta = timedelta(days_since[iday])
    offset = start_date + delta
    dates_since = np.append(dates_since,offset)
    years = np.append(years,dates_since[iday].year)
    months = np.append(months,dates_since[iday].month)
    days = np.append(days,dates_since[iday].day)
    
year_vec = np.arange(1978,2020) #years spanning
dates_in_yr = np.array([])
for iyr in np.arange(1,len(year_vec)):
    yrs_ind = np.where(years == year_vec[iyr])
    days_per_yr = np.array(yrs_ind).flatten()
    dates_in_yr = np.append(dates_in_yr,len(days_per_yr))
    

#select months
    
mon_select = [6,7,8,9]
SIC_select = np.array([])
for imon in np.arange(0,len(mon_select)):
    mon_i = np.where(months == mon_select[imon])
    SIC_select = np.append(SIC_select,SIC[mon_i])

SIC_select_mean = np.nanmean(SIC_select)
SIC_select_anom = SIC_select - np.nanmean(SIC_select)

SIC_hist,xbins,_ = plt.hist(SIC_select,bins=np.linspace(0,15),histtype='step')
bin_centers = 0.5*(xbins[1:]+xbins[:-1])

fig1 = plt.figure()
ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(SIC)
plt.xticks(dates_in_yr,year_vec)
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
ax2.plot(SIC_select)
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])
ax3.plot(bin_centers,SIC_hist)
plt.show()

fig4 = plt.figure()
sns.distplot(np.divide(SIC_select,len(SIC_select)),hist=False,kde=True,bins=15,color='black',kde_kws={'linewidth': 4})
