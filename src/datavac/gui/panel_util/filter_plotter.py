from enum import Enum
from typing import Union, Any

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from pandas import DataFrame
from panel.widgets import Widget, MultiSelect, Select
import panel as pn
import param as hvparam
import pandas as pd

from datavac.gui.bokeh_util.util import make_serializable
from datavac.gui.panel_util.inst_params import CompositeWidgetWithInstanceParameters
from datavac.io.hose import Hose
from datavac.logging import logger

class SelectionHint(Enum):
    FIRST=1

class FilterPlotter(CompositeWidgetWithInstanceParameters):
    _prefilter_measgroup = None
    _sources:list[ColumnDataSource] =[]
    _pre_sources:list[DataFrame]=[]

    _prefilter_updated_count=hvparam.Event()
    _filter_param_updated_count=hvparam.Event()
    _need_to_recreate_figure=hvparam.Event()
    filter_settings=hvparam.Parameter()
    view_settings=hvparam.Parameter()
    meas_groups=hvparam.Parameter()

    def __init__(self, hose:Hose=None, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self._pre_filters:dict[str,Any]={}
        self.set_hose(hose)

        self._filter_param_widgets = self._make_filter_params(self.filter_settings)
        self._view_param_widgets = self._make_view_params(self.view_settings)
        self.update_sources(pre_sources=None)
        self._fig=self.recreate_figures
        self._composite[:]=[self._make_layout(
                                    filter_widgets=self._filter_param_widgets,
                                    view_widgets=self._view_param_widgets,
                                    figure_pane=self._fig)]
        self.param.watch(self._prefilter_updated,'_prefilter_updated_count')
        self.param.watch(self._filter_param_updated,'_filter_param_updated_count')
        self.param.watch((lambda *args,**kwargs:self.param.trigger('_filter_param_updated_count')),list(self._filter_param_widgets.keys()))
        # ?
        self.param.watch((lambda *args,**kwargs:self.update_sources_and_figures()),list(self._view_param_widgets.keys()))

    def _make_filter_params(self,filter_settings):
            return self._make_params_from_settings(filter_settings)
    def _make_view_params(self,view_settings):
            return self._make_params_from_settings(view_settings)

    def _make_params_from_settings(self,settings):
        widgets={}
        for param_name,elt in settings.items():
            if type(elt) is list:
                self.param.add_parameter(param_name,hvparam.ListSelector())
                widgets[param_name]=MultiSelect.from_param(
                    self.param[param_name],  sizing_mode='stretch_both')
            elif isinstance(elt,hvparam.Parameter):
                self.param.add_parameter(param_name,elt)
                if isinstance(elt,hvparam.ListSelector):
                    widgets[param_name]=MultiSelect.from_param(self.param[param_name], sizing_mode='stretch_both')
                elif isinstance(elt,hvparam.Selector):
                    widgets[param_name]=Select.from_param(self.param[param_name], sizing_mode='stretch_both')
            else:
                raise NotImplementedError
        return widgets

    @staticmethod
    def _make_layout(filter_widgets, view_widgets, figure_pane):
        top_row=pn.Row(*filter_widgets.values(),sizing_mode='stretch_width',height=125)
        main_row=pn.Row(figure_pane,*view_widgets.values(),sizing_mode='stretch_width',height=300)
        return pn.Column(top_row,main_row,sizing_mode='stretch_both')

    # If you make a pane out of this function, be sure to put in a "@pn.depends()"
    @pn.depends('_need_to_recreate_figure')
    def recreate_figures(self):
        fig=figure(width=400,height=300,sizing_mode='fixed')
        fig.line(x=[1,2,3],y=[1,2,3])
        return fig

    def set_hose(self,hose:Hose):
        self._hose=hose

    def add_prefilters(self, pre_filters, prefilter_measgroup):
        assert self._prefilter_measgroup is None or self._prefilter_measgroup == prefilter_measgroup
        self._prefilter_measgroup=prefilter_measgroup

        with hvparam.parameterized.batch_call_watchers(self):
            for k, item in pre_filters.items():
                if k in self._pre_filters:
                    raise Exception(f"Already have a prefilter for {k}")
                if isinstance(item,Widget):
                    item.param.watch((lambda *args,**kwargs:
                          self.param.trigger('_prefilter_updated_count')),'value')
                    if item.value!=[]: self.param.trigger('_prefilter_updated_count')
            self._pre_filters.update(pre_filters)

    def _prefilter_updated(self, *args, **kwargs):
        # Get all the pre_filters
        pre_filter_params={}
        for k,item in self._pre_filters.items():
            if type(item) in [list,tuple]:
                pre_filter_params[k]=item if item != [] else ['None']
            else:
                pre_filter_params[k]=item.value if item.value != [] else ['None']

        logger.debug(f"Pre-filter params: {pre_filter_params}")
        # Get the factors of pre_filtered data
        with hvparam.parameterized.batch_call_watchers(self):
            for param in self.filter_settings:
                all_factors=self._hose.get_factors(
                    self._prefilter_measgroup, param, **pre_filter_params)
                factors=[f for f in all_factors if f==f]
                if len(factors)!=len(all_factors):
                    logger.debug(f"Excluding {[f for f in factors if f!=f]} from factors for {param}")
                getattr(self.param,param).objects=factors
                #logger.debug(f"Update factors for {param} from {getattr(self.param,param).objects} to {factors}")
                acceptables=[v for v in (getattr(self,param) or []) if v in factors]
                if acceptables==[] and len(factors):
                    if self.filter_settings[param]==SelectionHint.FIRST:
                        acceptables=[factors[0]]
                    else:
                        acceptables=[v for v in self.filter_settings[param] if v in factors]
                if (getattr(self,param) or [])!=acceptables:
                    setattr(self,param,acceptables)
            self.param.trigger('_filter_param_updated_count')

    def _filter_param_updated(self,event):
        if event.type!="triggered":
            return
        self._update_data()
        self.update_sources_and_figures()

    def update_sources(self,pre_sources):
        raise NotImplementedError

    def _update_data(self):
        logger.info(f"Update data for {type(self).__name__}")
        factors={param:getattr(self,param) for param in self.filter_settings}
        factors.update({param:w.value for param,w in self._pre_filters.items()})
        #if not len(factors.get('LotWafer',[]) or []):
        #    if not len(self.lots_preselection):
        #        factors['LotWafer']=['None']
        #    else:
        #        factors['LotWafer']=[lw for lw in self.param.LotWafer.objects
        #                             if lw.split("_")[0] in self.lots_preselection]
        #sort_by=getattr(self,self._sort_field_param) if self._sort_field_param else None
        #data=self.preprocess_data(self.fetch_data(factors=factors,sort_by=sort_by))
        self._pre_sources=self.preprocess_data(self.fetch_data(factors=factors,sort_by='None'))
        self._dtypes=[d.dtypes.to_dict() for d in self._pre_sources]


        #logger.info("About to update _source")
        #if refashion_source or (self._sources is None):
        #    logger.debug("Refashioning source")
        #    self._sources=[ColumnDataSource(d) for d in data]
        #else:
        #    for source,d in zip(self._sources,data): source.data=d
        #self._updated_data_count+=1
        #logger.info("Did update data")
        #self._post_update_data()


    def fetch_data(self, factors, sort_by):
        scalar_columns = self.get_scalar_column_names()
        raw_columns = self.get_raw_column_names()
        logger.info(f"About to ask hose")  # , scalar: {scalar_columns}, raw: {raw_columns}")
        data: list[DataFrame] = [self._hose.get_data(meas_group,
                                                     scalar_columns=sc, raw_columns=rc, on_missing_column='none',
                                                     **factors)
                                 for sc, rc, meas_group
                                 in zip(scalar_columns, raw_columns, self.meas_groups)]
        logger.info("Got data from hose")

        for d in data:
            if sort_by in d.columns:
                d.sort_values(by=sort_by, inplace=True)

        return data

    def preprocess_data(self,predata):
        logger.debug(f"Default preprocess {self.__class__}")
        #print(predata[-1])
        #return predata[pd.notna(predata).all(axis=1)]
        #data=predata.where(predata.notna(),None)
        data=[]
        rcs=self.get_raw_column_names()
        for pred,rc in zip(predata,rcs):
            d=make_serializable(pred)
            for c in rc:
                if c not in d:
                    d[c]=[[]]*len(data)
                try:
                    d.loc[:,c]=d[c].where(~pd.isna(d[c]),pd.Series([[]]*len(d),dtype='object'))
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
                    print('oops')
                    raise
            data.append(d)
            #print("Post-processing dtypes:",data.dtypes)
        return data
    def polish_figures(self):
        pass

    def update_sources_and_figures(self):
        self.update_sources(self._pre_sources)
        self.polish_figures()