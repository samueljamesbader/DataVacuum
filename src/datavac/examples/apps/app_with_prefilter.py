from functools import reduce

import panel as pn
from panel.template.base import BasicTemplate
import param as hvparam

from datavac import logger
from datavac.appserve.app import PanelApp
from datavac.gui.panel_util.filter_plotter import FilterPlotter
from datavac.gui.panel_util.selectors import VerticalCrossSelector
from datavac.io.database import get_database

class PanelAppWithLotPrefilter(PanelApp,hvparam.Parameterized):
    hose_source: str
    hose_analysis: str
    lot_prefetch_measgroup: str
    _need_to_update_lot_filter=hvparam.Event()
    def __init__(self, plotters: dict[str, FilterPlotter],*args,**kwargs):
        super().__init__()
        hvparam.Parameterized.__init__(self,*args,**kwargs)
        self.database=get_database()
        self.lots_preselector=VerticalCrossSelector(
            value=[], options=[], width=170, height=500)
        self._plotters:dict[str,FilterPlotter]=plotters
        for name, pltr in self._plotters.items():
            pltr.add_prefilters({'Lot':self.lots_preselector},self.lot_prefetch_measgroup)
        self.param.watch(self._populate_lots,['_need_to_update_lot_filter'])
        self.param.trigger('_need_to_update_lot_filter')

    async def _populate_lots(self,*args,**kwargs):
        #self.lots_preselector.options=self.hose.get_lots(self.lot_prefetch_measgroup)
        self.lots_preselector.options=self.database.get_factors(self.lot_prefetch_measgroup,factor_names=['Lot'])['Lot']

    def get_page(self) -> BasicTemplate:
        self.page.sidebar.append(pn.panel("## Lot pre-filter"))
        self.page.sidebar.append(self.lots_preselector)
        self.page.sidebar.append(pn.HSpacer(height=20))
        self.page.sidebar.append(pn.widgets.FileDownload(callback=self._download_callback,filename=f'{self.title} Download.csv',label='Download shown'))
        self.page.sidebar_width=220

        tabs=[]
        for name, pltr in self._plotters.items():
            tabs.append((name, pltr))
        self._tabs=pn.Tabs(*tabs)
        self.page.main.append(self._tabs)
        self.page.main.sizing_mode='fixed'
        return self.page

    def link_shared_widgets(self):
        attrs=['_filter_param_widgets','_view_param_widgets']
        for attr in attrs:
            all_filter_params=set([k for p in self._plotters.values() for k in getattr(p,attr)])
            for k in all_filter_params:
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