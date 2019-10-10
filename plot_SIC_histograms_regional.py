import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from scipy import signal
import scipy.stats as stats
import pandas as pd
# Marie C. McGraw
# Atmospheric Science, University of Washington
# Updated 07-16-2019
#
# This script plots the SIC anomalies and change in SIC anomaly for each region (defined by NSIDC) from 1989-2018. 

#load observed NSIDC Sea Ice Extent ('Extent') that has been regridded on sipn grid and divided into NSIDC regions   
filepath = '/home/disk/sipn/nicway/data/obs/NSIDC_0079/sipn_nc_yearly_agg/'  #observed sea ice extent data
##open_mfdataset(foo+'/*.nc') opens all nc files in specific directory
filename = xr.open_mfdataset(filepath+'/*.nc',concat_dim='time')
region_names = filename.region_names #list of region names
SIC_load = filename.Extent #sea ice extent for all regions
#Create Barents-Kara Sea [7-8], Laptev-East Siberian Sea [9-10], Beaufort-Chukchi Sea [11-12]
SIC_load = np.column_stack((SIC_load,(SIC_load[:,7]+SIC_load[:,8]),
                            (SIC_load[:,8]+SIC_load[:,9]),
                            (SIC_load[:,10]+SIC_load[:,11]+SIC_load[:,12])))
region_names = np.append(region_names,('Barents-Kara',
                                       'Kara-Laptev',
                                       'East-Siberian-Beaufort-Chukchi'))

#get time (in datetime format)
time2 = filename.time
months = np.array(time2.dt.month)
days = np.array(time2.dt.day)
years = np.array(time2.dt.year)
TIME = np.transpose(np.stack([years,months,days]))

#read in SIC.  It is a time series for each region. Plot the time series, and the time series with its mean removed
no_regions = len(region_names)
DETREND = False
mon_sel = [1,2,3,4,5,6,7,8,9,10,11,12] #selected months
mon_sel_str = 'ALL' #string of selected months (for filenames)
day_sep = 2 #days +/- the center date
no_days = 2*day_sep + 1

#i = 12
for i in np.arange(0,no_regions):
    #Make sure region name is a string; create a no spaces version for filenames
    region_i = region_names[i]
    #region_i_fn = np.array2string(region_i)
    region_i_fn = region_i.replace(" ","") #replace spaces in names with nothing
    print('now running {region_name}'.format(region_name=region_i))
    SIC = SIC_load[:,i]
    #Remove mean
    SIC_mean_rem = SIC - np.nanmean(SIC)
    #Remove linear trend
    if DETREND == True:
        SIC_dt = signal.detrend(SIC)
#        pf = np.polyfit(np.arange(0,len(SIC)),SIC_mean_rem,1)
#        polyval = np.poly1d(pf)
#        #SIC_dt = SIC_mean_rem - polyval(np.arange(0,len(SIC)))
#        SIC_dt = polyval(np.arange(0,len(SIC)))
        detrend_stat = 'dt'
    else:
        SIC_dt = SIC_mean_rem
        detrend_stat = 'NO_dt'
        
    #plot initial SIC, mean removed, and detrended
    fig1, (ax1a, ax1b) = plt.subplots(1,2, figsize=(15,5),tight_layout = True)
    ax1a.plot(SIC)
    ax1a.set_xlabel('Time')
    ax1a.set_ylabel('SIC (10^6 km^2)')
    ax1a.set_title('Sea Ice Extent from NSIDC, {region_name}, 1989-2018'.format(region_name=region_i))
    #
    ax1b.plot(SIC_mean_rem,'r')
    ax1b.plot(SIC_dt,'k')
    ax1b.set_xlabel('Time')
    ax1b.set_ylabel('SIC (10^6 km^2)')
    ax1b.legend(['mean removed','detrended'])
    ax1b.set_title('Sea Ice Extent from NSIDC, {region_name}, 1989-2018'.format(region_name=region_i))
    #save
    fname_1 = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/diagnostic_figures/SIC_raw_{detrend_stat}_{region_name}'.format(detrend_stat=detrend_stat,region_name=region_i_fn)
    #plt.show()
    plt.savefig(fname_1,format='png',dpi=600)
    
    
    #Remove seasonal cycle
    N = len(TIME) #number of time steps
    SIC_seasonal_cycle = np.array([])
    for iseas in np.arange(0,N):
        idate = TIME[iseas,:]
        ind_sel = np.where((TIME[:,1] == idate[1]) & ((idate[2]-day_sep <= TIME[:,2]) & (TIME[:,2] <= idate[2]+2)))
        SIC_seasonal_cycle = np.append(SIC_seasonal_cycle,(SIC_dt[iseas,] - np.nanmean(SIC_dt[ind_sel,])))
    
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
    ax2.plot(SIC_dt)
    ax2.plot(SIC_seasonal_cycle,'k--')
    ax2.set_xlabel('Time (1989-2018)')
    ax2.set_ylabel('SIC (10^6 km^2)')
    ax2.legend(['with seasonal cycle','seasonal cycle removed'])
    ax2.set_title('SIC, seasonal cycle removed, {region_name}, 1989-2018'.format(region_name=region_i))
    fname_2 = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/diagnostic_figures/SIC_seas_cyc_rem_{detrend_stat}_{region_name}'.format(detrend_stat=detrend_stat,region_name=region_i_fn)
    #plt.show()
    plt.savefig(fname_2,format='png',dpi=600)
    
    
    #Select only desired months
    mon_sel_ind = np.isin(months[0:len(TIME)],mon_sel)
    SIC_mon_sel = SIC_seasonal_cycle[np.where(mon_sel_ind==True)]
    fig3 = plt.figure()
    ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])
    ax3.plot(SIC_mon_sel)
    ax3.set_xlabel('Time (1989-2018)')
    ax3.set_ylabel('SIC (10^6 km^2)')
    ax3.set_title('{seas} SIC, seasonal cycle removed, {region_name}, 1989-2018'.format(seas=mon_sel_str,region_name=region_i))
    fname_3 = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/diagnostic_figures/{season}_SIC_seas_cyc_rem_{detrend_stat}_{region_name}'.format(season=mon_sel_str,detrend_stat=detrend_stat,region_name=region_i_fn)
    #plt.show()
    plt.savefig(fname_3,format='png',dpi=600)
    
    #Now make the histograms!
    fig4 = plt.figure()
    ax4 = fig4.add_axes([0.1,0.1,0.8,0.8])
    ax4.hist(SIC_mon_sel,bins=np.arange(-4,4,0.1),histtype=u'step',density=True,linewidth=3)
    ax4.set_xlabel('SIC (10^6 km^2)')
    ax4.set_ylabel('Relative frequency')
    ax4.set_title('{seas} SIC anomalies, {region_name}'.format(seas=mon_sel_str,region_name=region_i))
    fname_4 = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/SIC_anoms_{region_name}_{season}_hist_{detrend_stat}'.format(region_name=region_i_fn,season=mon_sel_str,detrend_stat=detrend_stat)
    #plt.show()
    plt.savefig(fname_4,format='png',dpi=600)
    
    #Histogram of change in SIC
    SIC_delta = np.array([])
    for idelta in np.arange(day_sep,len(SIC_seasonal_cycle)-day_sep):
        SIC_idelta = SIC_seasonal_cycle[idelta+day_sep] - SIC_seasonal_cycle[idelta-day_sep]
        SIC_delta = np.append(SIC_delta,SIC_idelta)
    
    SIC_delta_sel = SIC_delta[np.where(mon_sel_ind[day_sep:len(mon_sel_ind)-day_sep] == True)]
    
    fig5 = plt.figure()
    ax5 = fig5.add_axes([0.1,0.1,0.8,0.8])
    ax5.hist(SIC_delta_sel,bins=np.arange(-0.3,0.3,0.01),histtype=u'step',density=False,linewidth=3)
    ax5.set_xlabel('{no_days} day change in SIC, 1989-2018 (10^6 km^2)'.format(no_days = no_days))
    ax5.set_ylabel('count')
    ax5.set_title('{seas} change in {no_days}-day SIC anomalies, {region_name}'.format(seas=mon_sel_str,no_days=no_days,region_name=region_i))
    fname_5 = '/home/disk/sipn/mcmcgraw/figures/VRILE/Zhou_Wang_replicate/figures_results/SIC_anoms_{no_days}day_change_{region_name}_{season}_hist_{detrend_stat}'.format(no_days=no_days,region_name=region_i_fn,season=mon_sel_str,detrend_stat=detrend_stat)
    #plt.show()
    plt.savefig(fname_5,format='png',dpi=600)
    
    plt.close('all')
    
    ##Save change in SIC to text file
    fname_data_save = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS/NSIDC_SIE_delta_{nodays}day_change_{region_name}_{seas}_{detrend_stat}.txt'.format(nodays=no_days,region_name=region_i_fn,seas=mon_sel_str,detrend_stat=detrend_stat)
    np.savetxt(fname_data_save,np.array(SIC_delta_sel),fmt='%.9f')
    
fname_data_save_TIME = '/home/disk/sipn/mcmcgraw/python/data_VRILEs/text_files/OBS/NSIDC_SIE_delta_TIME_{nodays}day_change_{seas}_{detrend_stat}.csv'.format(nodays=no_days,seas=mon_sel_str,detrend_stat=detrend_stat)
TIME_pd = pd.DataFrame(TIME,columns=['year','month','day'])
TIME_pd.to_csv(fname_data_save_TIME)


    
    

    

