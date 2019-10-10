import cdsapi
import numpy as np
#yr_vec = [1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 
#          2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
#          2015, 2016]
yr_vec = [1993,1994]
mon_vec = ('01','02','03','04','05','06','07','08','09','10','11','12')
c = cdsapi.Client()

for i in yr_vec:
    for imon in np.arange(0,12):
#i = 1993
        c.retrieve(
            'seasonal-original-pressure-levels',
            {
                'format':'netcdf',   
                'originating_centre':'ecmwf',
                'system':'5',
                'variable':'geopotential',
                'pressure_level':'500',
                'year':str(i),
                'month':mon_vec[imon],
                'day':'01',
                'leadtime_hour':[
                    '24', '48', '72', '96', '120', '144', 
                    '168', '192', '216', '240', '264', '288', 
                    '312', '336', '360', '384', '408', '432', 
                    '456', '480', '504', '528', '552', '576', 
                    '600', '624', '648', '672', '696', '720'
                ],
                #'format':'grib'
            },
            'download_{yr}_{mon}_test.nc'.format(yr=i,mon=imon+1))
