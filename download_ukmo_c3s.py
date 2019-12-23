#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:01:35 2019

@author: mcmcgraw
"""

import cdsapi
yrs_save = '1993_1994'
c = cdsapi.Client()

c.retrieve(
    'seasonal-original-pressure-levels',
    {
        'originating_centre': 'ukmo',
        'system': [
            '12', '13', '14',
        ],
        'variable': 'geopotential',
        'pressure_level': '500',
        'year': [
            '1993', '1994',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '09', '17',
            '25',
        ],
        'leadtime_hour': [
            '24', '48', '72',
            '96', '120', '144',
            '168', '192', '216',
            '240', '264', '288',
            '312', '336', '360',
            '384', '408', '432',
            '456', '480', '504',
            '528', '552', '576',
            '600', '624', '648',
            '672', '696', '720',
        ],
        'format': 'netcdf',
    },
    '{yrs_save}.nc'.format(yrs_save=yrs_save))

