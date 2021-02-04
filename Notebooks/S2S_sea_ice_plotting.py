import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import seaborn as sns
import os
import glob2

#1. Plot VRILE counts as a function of month AND model for each region.  data_plt should contain the model data and any desired obs
def VRILE_count_plot(ax_plt,data_plt,reg_sel,TO_PLOT,iplt,v_min,v_max):
    # 
    panel_lett = ['a','b','c','d','e','f']
    #
    pd_plt = pd.pivot_table(data=data_plt,index='model name',columns='valid month',
                             values=TO_PLOT,aggfunc=np.mean,dropna=False)
    # heatmap
    hmap = sns.heatmap(pd_plot,ax=ax_plt,center=0,vmin=vmin,vmax=vmax,annot=True,linewidth=1.3,annot_kws={"size": 13},
                linecolor='xkcd:gray',cmap='PuOr')#,cbar_ax=cbar_ax)
    #
    ax_plt.set_ylabel('',fontsize=22,rotation=90)
    ax_plt.set_xlabel('Forecast Valid Month',fontsize=20)
    ax_plt.set_xticks(np.arange(0.5,12.5))
    ax_plt.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontsize=18)
    #
    ax_plt.set_title('{i_lett} {model_name}'.format(i_lett=panel_lett[iplt],model_name=reg_sel),fontsize=20)
    ax_plt.collections[0].colorbar.set_label('%',rotation=0,fontsize=18,y=-0.04,labelpad=-20)
    #
    ax_plt.set_yticklabels(hmap.get_ymajorticklabels(),fontsize=18,rotation=0)
                     