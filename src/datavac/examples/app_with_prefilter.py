import panel as pn
from panel.template.base import BasicTemplate
import param as hvparam

from datavac.appserve.app import PanelApp
from datavac.gui.panel_util.filter_plotter import FilterPlotter
from datavac.gui.panel_util.selectors import VerticalCrossSelector
from datavac.io.hose import DBHose


class PanelAppWithLotPrefilter(PanelApp,hvparam.Parameterized):
    hose_source: str
    hose_analysis: str
    lot_prefetch_measgroup: str
    _need_to_update_lot_filter=hvparam.Event()
    def __init__(self,
                 plotters: dict[str, FilterPlotter]):
        super().__init__()
        hvparam.Parameterized.__init__(self)
        self.hose=DBHose(self.hose_source,self.hose_analysis)
        self.lots_preselector=VerticalCrossSelector(
            value=[], options=[], width=170, height=500)
        self._plotters=plotters
        for name, pltr in self._plotters.items():
            pltr.set_hose(self.hose)
            pltr.add_prefilters({'Lot':self.lots_preselector},self.lot_prefetch_measgroup)
        self.param.watch(self._populate_lots,['_need_to_update_lot_filter'])
        self.param.trigger('_need_to_update_lot_filter')

    async def _populate_lots(self,*args,**kwargs):
        self.lots_preselector.options=self.hose.get_lots(self.lot_prefetch_measgroup)

    def get_page(self) -> BasicTemplate:
        self.page.sidebar.append(pn.panel("## Lot pre-filter"))
        self.page.sidebar.append(self.lots_preselector)
        self.page.sidebar_width=220

        tabs=[]
        for name, pltr in self._plotters.items():
            tabs.append((name, pltr))
        self.page.main.append(pn.Tabs(*tabs))
        self.page.main.sizing_mode='fixed'
        return self.page