#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:00:55 2019

@author: mcmcgraw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:48:13 2019

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
import random
import matplotlib.colors as colors

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
day_change = 5
max_lead = 30

#Load counts
#Create custom colorbar
cmap_bounds = np.array([0,2.5,5,10,25,40,60,75,90,100])
color_list = ['#f97306','#6a79f7','#3f9b0b','#fd3c06','#bf77f6',
              '#cb416b','#929591','#fac205','#02d8e9']
custom_cmap = colors.ListedColormap(color_list)
cust_norm = colors.BoundaryNorm(boundaries=cmap_bounds,ncolors=9)
p5_count = pd.read_csv(filepath+'COUNT_ecmwfsipn_reforecast_stronger_than_obs_5th_pctile_RESAMPLED_MC.csv')
mon_labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
reg_names = p5_count.loc[:,'Region']
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])
actual_values = p5_count.iloc[:,1:13]
actual_values_plot = round(actual_values/10)*10
cp1 = ax1.pcolormesh(actual_values_plot,cmap=custom_cmap,norm=cust_norm,edgecolors='k')
cbar1 = plt.colorbar(cp1,ticks=cmap_bounds,pad=0.05)
cbar1.ax.set_ylabel('%')
ax1.set_xticks(np.arange(0.5,12.5,1))
ax1.set_xticklabels(mon_labels)
ax1.set_yticks(np.arange(0.5,17.5,1))
ax1.set_yticklabels(reg_names,fontsize=9)
ax1.set_title('How Often is the Modeled VRILE Stronger than the Observed VRILE?')
fname_1 = '/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_modeled_VRILE_stronger_than_observed.png'.format(model_name=model_name,
                                                  model_type=model_type)
fig1.savefig(fname_1,format='png',dpi=600,bbox_inches='tight')
#Same for 95th pctile
p95_count = pd.read_csv(filepath+'COUNT_ecmwfsipn_reforecast_stronger_than_obs_95th_pctile_RESAMPLED_MC.csv')
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
actual_values_p95 = p95_count.iloc[:,1:13]
actual_values_p95_plot = round(actual_values_p95/10)*10
cp2 = ax2.pcolormesh(actual_values_p95_plot,cmap=custom_cmap,norm=cust_norm,edgecolors='k')
cbar2 = plt.colorbar(cp2,ticks=cmap_bounds,pad=0.05)
cbar2.ax.set_ylabel('%')
ax2.set_xticks(np.arange(0.5,12.5,1))
ax2.set_xticklabels(mon_labels)
ax2.set_yticks(np.arange(0.5,17.5,1))
ax2.set_yticklabels(reg_names,fontsize=9)
ax2.set_title('How Often is the Modeled VRIGE Stronger than the Observed VRIGE?')
fname_2 = '/home/disk/sipn/mcmcgraw/figures/VRILE/{model_name}_{model_type}_modeled_VRIGE_stronger_than_observed.png'.format(model_name=model_name,
                                                  model_type=model_type)
fig2.savefig(fname_2,format='png',dpi=600,bbox_inches='tight')