import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_SIE_forecast_example_plot(DATA,OBS,model_names_ALL,TO_PLOT,ax_sel,is_LEGEND,colors):
    
    for imod in np.arange(0,len(model_names_ALL)):
        imod_name = model_names_ALL[imod]
        i_SIE_model_group = DATA.xs(imod_name,level=1)
        init_times_sel = i_SIE_model_group.index.get_level_values('init date').unique()
        if imod_name == 'NCEP':
            for itime in np.arange(0,len(init_times_sel),2):
                SIE_model_itime = i_SIE_model_group.xs(init_times_sel[itime],level=1).reset_index()
                sns.lineplot(data=SIE_model_itime,x='valid date',y=TO_PLOT,ax=ax_sel,linewidth=1,color=colors[imod],
                       alpha=0.5)
        else:
            for itime in np.arange(0,len(init_times_sel)):
    #itime = '2012-07-18'
                SIE_model_itime = i_SIE_model_group.xs(init_times_sel[itime],level=1).reset_index()
                sns.lineplot(data=SIE_model_itime,x='valid date',y=TO_PLOT,ax=ax_sel,linewidth=1,color=colors[imod],
                       alpha=0.5)

    ##
    for imod2 in np.arange(0,len(model_names_ALL)):
    #imod2 = 0
        mn2 = model_names_ALL[imod2]
        if mn2 != 'NCEP':
            day_0s = DATA.xs((slice(None),mn2,slice(None),0)).reset_index()
            day_0s_plt = day_0s.groupby(['valid date'])[TO_PLOT].mean()
            sns.scatterplot(data=day_0s_plt.reset_index(),x='valid date',y=TO_PLOT,ax=ax_sel,
                    s=180,color=colors[imod2],label=(mn2 if is_LEGEND==True else None))
        else:
            day_0s = DATA.xs((slice(None),mn2,slice(None),pd.Timedelta(1,'D'))).reset_index()
            day_0s_plt = day_0s.groupby(['valid date'])[TO_PLOT].mean()
            day_0s_plt = day_0s_plt.iloc[::2]
            sns.scatterplot(data=day_0s_plt.reset_index(),x='valid date',y=TO_PLOT,ax=ax_sel,
                s=180,color=colors[imod2],label=(mn2 if is_LEGEND==True else None))
    sns.lineplot(data=OBS.reset_index(),x='valid date',y=TO_PLOT,style='Model Name',color='k',ax=ax_sel,
             linewidth=3,legend=('full' if is_LEGEND == True else False))
    ax_sel.set_xlim([DATA.reset_index()['valid date'].min(),
               DATA.reset_index()['valid date'].max()])
    ax_sel.legend
    ax_sel.set_xlabel('Valid Date',fontsize=17)
    ax_sel.set_ylabel('Sea Ice Extent (10$^6$ km$^2$)',fontsize=17)
    ax_sel.grid()
    ax_sel.set_title('{TO_PLOT} Forecasts'.format(TO_PLOT=TO_PLOT),fontsize=19)
    
    #
def create_SIE_forecast_ERROR_example_plot(DATA,model_names_ALL,TO_PLOT,ax_sel,is_LEGEND,colors):
    sns.set_palette(colors,len(colors))
    for imod in np.arange(0,len(model_names_ALL)):
        imod_name = model_names_ALL[imod]
        i_SIE_model_group = DATA.xs(imod_name,level=1)
        init_times_sel = i_SIE_model_group.index.get_level_values('init date').unique()
        if imod_name == 'NCEP':
            for itime in np.arange(0,len(init_times_sel),2):
                SIE_model_itime = i_SIE_model_group.xs(init_times_sel[itime],level=0).reset_index()
                sns.lineplot(data=SIE_model_itime,x='valid date',y=TO_PLOT,ax=ax_sel,linewidth=1,color=colors[imod],
                       alpha=0.5)
        else:
            for itime in np.arange(0,len(init_times_sel)):
    #itime = '2012-07-18'
                SIE_model_itime = i_SIE_model_group.xs(init_times_sel[itime],level=0).reset_index()
                sns.lineplot(data=SIE_model_itime,x='valid date',y=TO_PLOT,ax=ax_sel,linewidth=1,color=colors[imod])
    #sns.lineplot(data=obs_SIE_region_sel.reset_index(),x='valid date',y=TO_PLOT,style='model name',color='k',ax=ax_sel,
    #         linewidth=3)
    for imod2 in np.arange(0,len(model_names_ALL)):
    #imod2 = 0
        mn2 = model_names_ALL[imod2]
        if mn2 != 'NCEP':
            day_0s = DATA.xs((slice(None),mn2,slice(None),0)).reset_index()
            day_0s_plt = day_0s.groupby(['valid date'])[TO_PLOT].mean()
            sns.scatterplot(data=day_0s_plt.reset_index(),x='valid date',y=TO_PLOT,ax=ax_sel,
                    s=180,color=colors[imod2],label=(mn2 if is_LEGEND==True else None))
        else:
            day_0s = DATA.xs((slice(None),mn2,slice(None),pd.Timedelta(1,'D'))).reset_index()
            day_0s_plt = day_0s.groupby(['valid date'])[TO_PLOT].mean()
            day_0s_plt = day_0s_plt.iloc[::2]
            sns.scatterplot(data=day_0s_plt.reset_index(),x='valid date',y=TO_PLOT,ax=ax_sel,
                s=180,color=colors[imod2],label=(mn2 if is_LEGEND==True else None))
    #
    ax_sel.set_xlim([DATA.reset_index()['valid date'].min(),
                   DATA.reset_index()['valid date'].max()])
    ax_sel.set_xlabel('Valid Date',fontsize=17)
    ax_sel.set_ylabel('Sea Ice Extent (10$^6$ km$^2$)',fontsize=17)
    ax_sel.grid()
    ax_sel.set_title('Error in {TO_PLOT} Forecasts'.format(TO_PLOT=TO_PLOT),fontsize=19)
    #sns.set_palette(sns.xkcd_palette(colors),len(colors))

    ax_sel.axhline(0,color='k')
### Calculate bias as a function of month and model for a specified lead time
def S2S_bias_plot(DATA,TO_PLOT,vmin,vmax,ax_sel,imod,region_sel):
    # letters
    mon_labels = ['J','F','M','A','M','J','J','A','S','O','N','D']
    letters = ['a)','b)','c)','d)','e)','f)','g)']
    # Create pivot table
    piv_plt = pd.pivot_table(data=DATA,index='model name',columns='valid month',values=TO_PLOT,aggfunc=np.mean,
                            dropna=False)
    # Heatmap
    sns.heatmap(piv_plt,cmap = 'PuOr',ax=ax_sel,linewidth=0.2,linecolor='xkcd:slate grey',
                vmin=vmin,vmax=vmax,center=0)
    # formatting
    ax_sel.set_yticklabels(piv_plt.index,rotation=0,fontsize=14)
    ax_sel.set_xticklabels(mon_labels,fontsize=14,rotation=0)
    ax_sel.set_ylabel(None)
    ax_sel.set_xlabel(None)
    ax_sel.set_title('{lett} {region}'.format(lett=letters[imod],region=region_sel),fontsize=15)
    if (TO_PLOT == 'SIE') | (TO_PLOT == 'SIE clim'):
        ax_sel.collections[0].colorbar.set_label('10$^6$ km$^2$',rotation=0,fontsize=13,y=-0.04,labelpad=-20)
    elif (TO_PLOT == 'SIE pct') | (TO_PLOT == 'SIE clim pct'):
        ax_sel.collections[0].colorbar.set_label('%',rotation=0,fontsize=13,y=-0.04,labelpad=-20)
### Plot RMSE as a function of lead time for each model
def RMSE_plot(ax_plt,data_plt,reg_sel,TO_PLOT,iplt,v_min,v_max,no_regions,ncols,nrows):
    # subplot panel labels
    panel_lett = ['a','b','c','d','e','f',
                 'g','h','i','j','k','l']
    # Palette
    test_pal = ['crimson','cornflower','teal','purple','tangerine']
    sns.set_palette(sns.xkcd_palette(test_pal),5)
    # Line plot first
    sns.lineplot(data=data_plt,x='lead days',y=TO_PLOT,hue='Model Name',
                 ax=ax_plt,markers=False,linewidth=6,legend=False if iplt < no_regions - 1 else 'full')
    # Scatter plot on top
    sns.scatterplot(data=data_plt,x='lead days',y=TO_PLOT,hue='Model Name',
                             ax=ax_plt,legend=False,s=350)
    # 
    ax_plt.grid()
    ax_plt.set_ylim((v_min,v_max))
    ax_plt.tick_params(axis='x', labelsize=17)
    ax_plt.tick_params(axis='y', labelsize=17)
    if iplt >= ncols*(nrows-1):
        ax_plt.set_xlabel('Lead Time (Weeks)',fontsize=25)
    else:
        ax_plt.set_xlabel(None)
    if np.mod(iplt,ncols)==0:
        ax_plt.set_ylabel('RMSE (10$^6$ km$^2$)',fontsize=25)
    else:
        ax_plt.set_ylabel(None)
    if iplt == no_regions-1:
        leg = ax_plt.legend(bbox_to_anchor=(1.025,1),loc=2,borderaxespad=0,fontsize=28)
        for legob in leg.get_lines():
            legob.set_linewidth(6.0)
                      
    ax_plt.set_title('{lett}) {region}'.format(lett=panel_lett[iplt],region=reg_sel),fontsize=25)
    sns.despine(ax=ax_plt)
## 4. Plot VRILE counts as a function of month AND model for each region.  data_plt should contain the model data and any desired obs
def VRILE_count_plot(ax_plt,data_plt,reg_sel,TO_PLOT,iplt,vmin,vmax):
    # 
    panel_lett = ['a','b','c','d','e','f','g','h']
    #
    pd_plt = pd.pivot_table(data=data_plt,index='model name sort',columns='valid month',
                             values=TO_PLOT,aggfunc=np.mean,dropna=False)
    # heatmap
    hmap = sns.heatmap(pd_plt,ax=ax_plt,center=0,vmin=vmin,vmax=vmax,annot=True,linewidth=1.3,annot_kws={"size": 13},
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
#  Plot RMSE for error propagation
def RMSE_plot_error(ax_plt,data_plt,data_scatter,reg_sel,TO_PLOT,iplt,v_min,v_max,shift_days,clim_freq,seas_sel,no_regions,ncols,nrows):
    # subplot panel labels
    panel_lett = ['a','b','c','d','e','f',
                 'g','h','i','j','k','l']
    # Palette
    test_pal = ['cornflower','teal','purple','tangerine','crimson','burnt siena',]
    sns.set_palette(sns.xkcd_palette(test_pal),6)
    # Line plot first
    sns.lineplot(data=data_plt,x='lead days',y=TO_PLOT,hue='Model',style='type',
                 ax=ax_plt,markers=False,linewidth=3,legend=False if iplt < no_regions - 1 else 'full')
    # Scatter plot on top
    sns.scatterplot(data=data_scatter,x='lead days',y=TO_PLOT,hue='Model',style='type',
                             ax=ax_plt,legend=False,s=300)
    # 
    if clim_freq=='WEEKLY':
        shift_days_PLT = shift_days/7
    else:
        shift_days_PLT = shift_days
    ax_plt.axvline(x=shift_days_PLT,ymin=0,ymax=10,color='xkcd:charcoal',linewidth=4)
    vstart_txt = v_max-0.05
    if ((reg_sel == 'E. Sib./Beauf./Chuk. Sea')&(TO_PLOT == 'SIE RMSE')):
        vstart_txt = 1
    else:
        vstart_txt = v_max-0.05
    ax_plt.text(shift_days_PLT+0.25,vstart_txt,'VRILE Starts',fontsize=25)
    ax_plt.grid()
    ax_plt.set_ylim((v_min,v_max))
    ax_plt.tick_params(axis='x', labelsize=17)
    ax_plt.tick_params(axis='y', labelsize=17)
    if iplt >= ncols*(nrows-1):
        ax_plt.set_xlabel('Lead Time (Weeks)',fontsize=25)
    else:
        ax_plt.set_xlabel(None)
    if np.mod(iplt,ncols)==0:
        ax_plt.set_ylabel('RMSE (10$^6$ km$^2$)',fontsize=25)
    else:
        ax_plt.set_ylabel(None)
    if iplt == no_regions-1:
        leg = ax_plt.legend(bbox_to_anchor=(1.025,1),loc=2,borderaxespad=0,fontsize=30)
        for legob in leg.get_lines():
            legob.set_linewidth(6.0)              
    ax_plt.set_title('{lett}) {region}, {seas_sel}'.format(lett=panel_lett[iplt],region=reg_sel,seas_sel=seas_sel),fontsize=25)
    
    sns.despine(ax=ax_plt)