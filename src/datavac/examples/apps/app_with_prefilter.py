from functools import reduce
from typing import Optional, Sequence, Union

import panel as pn
from panel.io import hold
from panel.template.base import BasicTemplate
import param as hvparam

from datavac.util.logging import logger
from datavac.appserve.app import PanelApp
from datavac.gui.panel_util.filter_plotter import FilterPlotter
from datavac.gui.panel_util.selectors import VerticalCrossSelector
from datavac.io.database import get_database

class PanelAppWithLotPrefilter(PanelApp,hvparam.Parameterized):
    lot_prefetch_measgroup: Union[str,list]
    _need_to_update_lot_filter=hvparam.Event()
    def __init__(self, plotters: dict[str, FilterPlotter],initial_prefilter_lots:Optional[Sequence[str]]=None,*args,**kwargs):
        super().__init__()
        hvparam.Parameterized.__init__(self,*args,**kwargs)
        self.database=get_database()
        self.lots_preselector=VerticalCrossSelector(
            value=[], options=[], width=170, height=500)
        self._plotters:dict[str,FilterPlotter]=plotters
        lot_prefetch_measgroup=[self.lot_prefetch_measgroup]*len(self._plotters)\
            if type(self.lot_prefetch_measgroup) is str else self.lot_prefetch_measgroup
        for (name, pltr),pfmg in zip(self._plotters.items(),lot_prefetch_measgroup):
            pltr.add_prefilters({'Lot':self.lots_preselector},pfmg)
        self._initial_prefilter_lots=initial_prefilter_lots
        self.param.watch(self._populate_lots,['_need_to_update_lot_filter'])
        self.param.trigger('_need_to_update_lot_filter')

    async def _populate_lots(self,*args,**kwargs):
        #self.lots_preselector.options=self.hose.get_lots(self.lot_prefetch_measgroup)
        lot_prefetch_measgroup=[self.lot_prefetch_measgroup]*len(self._plotters) \
            if type(self.lot_prefetch_measgroup) is str else self.lot_prefetch_measgroup
        self.lots_preselector.options=sorted(self.database.get_factors(lot_prefetch_measgroup[0],factor_names=['Lot'])['Lot'],reverse=True)
        if self._initial_prefilter_lots is not None:
            self.lots_preselector.value=self._initial_prefilter_lots


    def get_page(self) -> BasicTemplate:
        self.page.sidebar.append(pn.panel("## Lot pre-filter"))
        self.page.sidebar.append(self.lots_preselector)
        self.page.sidebar.append(pn.HSpacer(height=20))
        self.page.sidebar.append(pn.widgets.FileDownload(callback=self._download_callback,filename=self._get_download_filename(),label='Download shown'))
        self.page.sidebar_width=220

        def make_tabs():
            tabs=[]
            for name, pltr in self._plotters.items():
                tabs.append((name, pltr))
            self._tabs=pn.Tabs(*tabs)
            self._tabs.param.watch(self._inform_plotters_of_tab_change,'active')
            self._inform_plotters_of_tab_change()
            return self._tabs
        self.page.main.append(pn.panel(make_tabs,defer_load=True))
        self.page.main.sizing_mode='fixed'
        return self.page

    def _inform_plotters_of_tab_change(self,*args,**kwargs):
        # Inform the newly visible plotter first since that's what the user will see
        logger.debug(f"Tab changed to {self._tabs.active}, which is {list(self._plotters.keys())[self._tabs.active]}")
        list(self._plotters.values())[self._tabs.active].visibility_changed(is_visible_now=True)

        # Then inform the others
        for i,p in enumerate(self._plotters.values()):
            if i!=self._tabs.active:
                p.visibility_changed(is_visible_now=False)

    def link_shared_widgets(self, except_params=[]):
        attrs=['_filter_param_widgets','_view_param_widgets']
        for attr in attrs:
            all_filter_params=set([k for p in self._plotters.values() for k in getattr(p,attr)])
            for k in all_filter_params:
                if k in except_params: continue
                widgets=[getattr(p,attr).get(k,None) for p in self._plotters.values()]
                widgets=[w for w in widgets if w is not None]
                if len(widgets)>1:
                    w1=widgets[0]
                    for w2 in widgets[1:]:
                        w1.jslink(w2,value='value',bidirectional=True)

    def _download_callback(self):
        logger.debug("In download callback")
        active_plotter=list(self._plotters.values())[self._tabs.active]
        return active_plotter.download_shown()
    def _get_download_filename(self):
        return 'DataVacuum Download.csv'
        #active_plotter=list(self._plotters.values())[self._tabs.active]
        #return active_plotter.get_download_filename()
