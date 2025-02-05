import numpy as np
import pandas as pd
import panel as pn
from panel.template.base import BasicTemplate

from datavac.appserve.app import PanelApp
from datavac.gui.bokeh_util.wafer import Waferplot
from datavac.io.diemap import get_die_geometry, get_die_table, get_custom_dieremap
from datavac.util.conf import CONFIG
from datavac.gui.bokeh_util.palettes import RdYlGn

class AppDieMapDisplay(PanelApp):
    def __init__(self):
        super().__init__()

    def get_page(self) -> BasicTemplate:
        tabs=[]
        for mask in CONFIG['array_maps']:
            subtabs=[]
            die_lb=get_die_geometry(mask)
            main_dietab=get_die_table(mask)
            main_dietab['DieLabel']=main_dietab['DieXY']
            main_dietab.loc[~main_dietab['DieXY'].isin(die_lb['complete_dies']),'DieLabel']=''
            main_dietab=main_dietab.set_index('DieXY')[['DieLabel','DieX','DieY','DieCenterA [mm]','DieCenterB [mm]']]
            main_dietab['color']='white'
            pltr=Waferplot(color='color',die_lb=die_lb,
                           pre_source=main_dietab,cmap={'white':'#FFFFFF'},fig_kwargs={'width':600,'height':600},
                           text='DieLabel')
            subtabs.append((f'Core: {mask}', pltr.fig))
            for name in CONFIG['custom_remaps'][mask]:
                crm=get_custom_dieremap(mask,name).drop(columns=['DieCenterA [mm]','DieCenterB [mm]'])
                dietab=pd.merge(main_dietab,crm,
                                left_on=['DieX','DieY'],right_on=['DieX','DieY'],
                                how='left',validate='1:1').set_index('DieXY')
                dietab['CustomDieLabel']=dietab['CustomDieX'].astype('str')+','+dietab['CustomDieY'].astype('str')
                dietab.loc[~dietab['DieLabel'].isin(die_lb['complete_dies']),'CustomDieLabel']=''
                pltr=Waferplot(color='color',die_lb=die_lb,
                               pre_source=dietab,cmap={'white':'#FFFFFF'},fig_kwargs={'width':600,'height':600},
                               text='CustomDieLabel')
                subtabs.append((f'Custom: {name}', pltr.fig))
            tabs.append((mask,pn.Tabs(*subtabs)))
        self._tabs=pn.Tabs(*tabs)
        self.page.main.append(self._tabs)
        self.page.main.sizing_mode='fixed'
        return self.page

if __name__=='__main__':
    AppDieMapDisplay().get_page().show()