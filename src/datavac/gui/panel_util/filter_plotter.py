from enum import Enum
from typing import Union, Any

import bokeh
from bokeh.models import ColumnDataSource, LinearColorMapper, LogColorMapper, FactorRange
from bokeh.plotting import figure
from bokeh.transform import jitter
from pandas import DataFrame
from panel.widgets import Widget, MultiSelect, Select, CompositeWidget
import panel as pn
import param as hvparam
import pandas as pd
import numpy as np
from param.parameterized import batch_call_watchers

from datavac.gui.bokeh_util import palettes
from datavac.gui.bokeh_util.util import make_serializable, make_color_col, smaller_legend
from datavac.gui.bokeh_util.wafer import waferplot, Waferplot
from datavac.gui.panel_util.inst_params import CompositeWidgetWithInstanceParameters
from datavac.io.hose import Hose
from datavac.logging import logger
from datavac.util import Normalizer


class SelectionHint(Enum):
    FIRST=1

def make_default_layout(filter_widgets, view_widgets, figure_pane):
    top_row=pn.Row(*filter_widgets.values(),sizing_mode='stretch_width',height=125)
    main_row=pn.Row(figure_pane,pn.Column(*view_widgets.values(),sizing_mode='stretch_height'),sizing_mode='stretch_width')
    return pn.Column(top_row,pn.HSpacer(height=10),main_row,sizing_mode='stretch_both')

class FilterPlotter(CompositeWidgetWithInstanceParameters):
    _prefilter_measgroup = None
    _sources:list[ColumnDataSource] =None
    _pre_sources:list[DataFrame]=None

    _built_in_view_settings=['color_by','norm_by']
    _prefilter_updated_count=hvparam.Event()
    _filter_param_updated_count=hvparam.Boolean(False)
    _need_to_recreate_figure=hvparam.Event()

    filter_settings=hvparam.Parameter()
    meas_groups=hvparam.Parameter()
    color_by=hvparam.Selector()
    norm_by=hvparam.Selector()

    normalization_details = hvparam.Parameter(instantiate=True)

    layout_function = hvparam.Callable(default=make_default_layout)

    def __init__(self, hose:Hose=None, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self._pre_filters:dict[str,Any]={}
        self.set_hose(hose)

        self._filter_param_widgets = self._make_filter_params(self.filter_settings)
        self._view_param_widgets = self._make_view_params({k:None for k in self._built_in_view_settings})
        self.update_sources(pre_sources=None)

        self._fig=self.recreate_figures
        self._composite[:]=[self.layout_function(
                                    filter_widgets=self._filter_param_widgets,
                                    view_widgets=self._view_param_widgets,
                                    figure_pane=self._fig)]

        self.param.watch(self._prefilter_updated,'_prefilter_updated_count')
        self.param.watch(self._filter_param_updated,'_filter_param_updated_count')
        self.param.watch((lambda *args,**kwargs:setattr(self,'_filter_param_updated_count',True)),
                         list(self._filter_param_widgets.keys()))
        self.param.watch(self.update_sources_and_figures,list(self._view_param_widgets.keys()))

        self._normalizer=Normalizer(self.normalization_details)
        self.param.norm_by.objects=self._normalizer.norm_options
        if self.norm_by is None: self.norm_by=self.param.norm_by.objects[0]

        if self.param.color_by.objects==[]:
            self.param.color_by.objects=list(self.filter_settings.keys())
        if self.color_by is None: self.color_by='LotWafer'

    def _make_filter_params(self,filter_settings):
        return self._make_params_from_settings(filter_settings)
    def _make_view_params(self,view_settings):
        return self._make_params_from_settings(view_settings)

    def _make_params_from_settings(self,settings):
        widgets={}
        for param_name,elt in settings.items():
            if (already_added:=(elt is None)):
                elt=self.param[param_name]
            if type(elt) is list or isinstance(elt,SelectionHint):
                if not already_added: self.param.add_parameter(param_name,hvparam.ListSelector())
                widgets[param_name]=MultiSelect.from_param(
                    self.param[param_name],  sizing_mode='stretch_both')
            elif isinstance(elt,hvparam.Parameter):
                if not already_added: self.param.add_parameter(param_name,elt)
                if isinstance(elt,hvparam.ListSelector):
                    widgets[param_name]=MultiSelect.from_param(self.param[param_name], sizing_mode='stretch_both')
                elif isinstance(elt,hvparam.Selector):
                    widgets[param_name]=Select.from_param(self.param[param_name], sizing_mode='stretch_width')
            else:
                raise NotImplementedError
        return widgets


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
                logger.debug(f"Update factors for {param}")
                all_factors=self._hose.get_factors(
                    self._prefilter_measgroup, param, **pre_filter_params)
                factors=[f for f in all_factors if f==f]
                if len(factors)!=len(all_factors):
                    logger.debug(f"Excluding {[f for f in factors if f!=f]} from factors for {param}")
                getattr(self.param,param).objects=factors
                #if not(len(factors)):
                #    setattr(self,param,None)
                logger.debug(f"Update factors for {param} from {getattr(self.param,param).objects} to {factors}")
                acceptables=[v for v in (getattr(self,param) or []) if v in factors]
                if acceptables==[] and len(factors):
                    if self.filter_settings[param]==SelectionHint.FIRST:
                        acceptables=[factors[0]]
                    else:
                        acceptables=[v for v in self.filter_settings[param] if v in factors]
                if (getattr(self,param) or [])!=acceptables:
                    setattr(self,param,acceptables)
            setattr(self,'_filter_param_updated_count',True)

    def _filter_param_updated(self,event):
        if not self._filter_param_updated_count: return
        self._filter_param_updated_count=False
        logger.info(f"Filter param updated {event.type} {event.new}")
        #if event.type!="triggered":
        #    return
        self._update_data()
        self.update_sources_and_figures(event)

    def update_sources(self,pre_sources):
        raise NotImplementedError

    def _update_data(self):
        logger.info(f"Update data for {type(self).__name__}")
        factors={param:getattr(self,param) for param in self.filter_settings}
        factors.update({param:w.value for param,w in self._pre_filters.items()})
        factors={k:v for k,v in factors.items() if v is not None}
        self._pre_sources=self.preprocess_data(self.fetch_data(factors=factors,sort_by='None'))
        self._dtypes=[d.dtypes.to_dict() for d in self._pre_sources]

    def get_raw_column_names(self):
        raise NotImplementedError

    def get_scalar_column_names(self):
        return [self._normalizer.normalizer_columns()+list(self.filter_settings.keys())]

    def get_meas_groups(self):
        return self.meas_groups

    def fetch_data(self, factors, sort_by):
        scalar_columns = self.get_scalar_column_names()
        raw_columns = self.get_raw_column_names()
        if any(v==[] for v in factors.values()):
            logger.info("Skipping DB call and return empty data because some factor is [].")
            return [pd.DataFrame({k:[] for k in scs+rcs}) for scs,rcs in zip(scalar_columns,raw_columns)]
        logger.info(f"About to ask hose with {factors}, scalar: {scalar_columns}, raw: {raw_columns}")
        data: list[DataFrame] = [self._hose.get_data(meas_group,
                                                     scalar_columns=sc, raw_columns=rc, on_missing_column='none',
                                                     **factors)
                                 for sc, rc, meas_group
                                 in zip(scalar_columns, raw_columns, self.get_meas_groups())]
        logger.info(f"Got data from hose, lengths {[len(d) for d in data]}")

        for d in data:
            if sort_by in d.columns:
                d.sort_values(by=sort_by, inplace=True)
        return data

    def preprocess_data(self,predata):
        logger.debug(f"Default preprocess {self.__class__}")
        data=[]
        rcs=self.get_raw_column_names()
        for pred,rc in zip(predata,rcs):
            d=make_serializable(pred)
            for c in rc:
                if c not in d:
                    d[c]=[np.array([])]*len(data)
                d[c]=d[c].map(np.asarray)
                try:
                    d.loc[:,c]=d[c].where(~pd.isna(d[c]),pd.Series([np.array([])]*len(d),dtype='object'))
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
                    print('oops')
                    raise
            data.append(d)
        return data
    def polish_figures(self):
        pass

    def update_sources_and_figures(self,event):
        logger.debug(f"Updating sources and figures because: {event.name}")
        self.update_sources(self._pre_sources)
        self.polish_figures()


class ScalarFilterPlotter(FilterPlotter):

    plot_pairs=hvparam.Parameter()
    fig_kwargs=hvparam.Parameter()
    plot_var_to_meas_group=hvparam.Parameter()
    merge_on_columns=hvparam.List()
    shownames=hvparam.Parameter()
    stars=hvparam.Parameter({})
    categoricals=hvparam.Parameter({})
    fig_arrangement=hvparam.Parameter('row')


    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # Set options and defaults for the view settings
        self.param.color_by.objects=list(self.filter_settings.keys())
        if self.color_by is None: self.color_by='LotWafer'

    @property
    def _meas_group_to_plot_vars(self):
        if not hasattr(self,'__meas_group_to_plot_vars'):
            assert (self.meas_groups is not None)!=(self.plot_var_to_meas_group is not None),\
                "Supply *either* meas_groups *or* plot_var_to_meas_group"
            if self.meas_groups is not None:
                assert len(self.meas_groups)==1, \
                    "If using more than one meas_group for ScalarFilterPlotter, supply plot_var_to_meas_group instead"
                meas_groups=self.meas_groups
                plot_vars=[[pi for pp in self.plot_pairs for p in pp for pi in ([p] if type(p) is str else p)]]
            if self.plot_var_to_meas_group is not None:
                assert self.merge_on_columns is not None,\
                    "If using more than one meas_group, gotta define how to merge them with merge_on_columns"
                meas_groups=sorted(self.plot_var_to_meas_group.values())
                plot_vars=[[pi for pp in self.plot_pairs for p in pp for pi in ([p] if type(p) is str else p)
                                    if self.plot_var_to_meas_group[p]==m]\
                               for m in meas_groups]
            self.__meas_group_to_plot_vars=dict(zip(meas_groups,plot_vars))
        return self.__meas_group_to_plot_vars

    def get_meas_groups(self):
        return list(self._meas_group_to_plot_vars.keys())

    def get_scalar_column_names(self):
        filter_and_norm_cols=super().get_scalar_column_names()[0]
        return [(pv+filter_and_norm_cols) for pv in self._meas_group_to_plot_vars.values()]

    def get_raw_column_names(self):
        return [[] for m in self._meas_group_to_plot_vars]

    def update_sources(self,pre_sources):
        if len(self.get_meas_groups())!=1:
            # Haven't implemented merging multiple tables yet
            # Could still make use of current class if every plot pair is within a table, but haven't implemented that yet
            raise NotImplementedError

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'data':ColumnDataSource({}),'stars':ColumnDataSource({})}

            # Empty prototype
            self._sources['data'].data=dict(**{str(p):[] for pp in self.plot_pairs for p in pp},**{'legend':[],'color':[]})
            self._sources['stars'].data=dict(**{str(p):[] for pp in self.plot_pairs for p in pp})
            self._sources['labels']=None
            #self._sources['data'].data={"('Lot', 'Wafer')":[('A','a'),('B','b')],'Cmean [F]':[1,2],'legend':['hi','sam'],'color':['red','blue']}

        # Otherwise, analyze the real data
        else:
            assert len(pre_sources)==1

            # Compile it all to right columns
            try:
                self._sources['data'].data=dict(
                    **{
                        str(p):(self._normalizer.get_scaled(pre_sources[0],p,self.norm_by) if p not in self.categoricals
                               else pre_sources[0][p] if type(p) is not tuple else list(zip(*[pre_sources[0][pi] for pi in p])))
                            for pp in self.plot_pairs for p in pp
                    },**{
                        'legend':pre_sources[0][self.color_by],
                        'color':make_color_col(pre_sources[0][self.color_by],
                                               all_factors=self.param[self.color_by].objects)
                    })
            except:
                print('oops')
                raise
            self._sources['stars'].data=dict(
                **{
                    p:[self._normalizer.get_scaled(self.stars,p,self.norm_by)]
                    for pp in self.plot_pairs for p in pp
                        if (p in self.stars and (self.norm_by=='None' or self.norm_by in self.stars))
                })
            self._sources['labels']={}
            for pp in self.plot_pairs:
                for p in pp:
                    eu=self._normalizer.formatted_endunits(p,self.norm_by)
                    sh=self._normalizer.shorthand(p,self.norm_by)
                    eupart=fr"\text{{ [{eu}]}}" if eu!="" else ""
                    self._sources['labels'][p]=fr"$${self.shownames[p]}{sh}{eupart}$$"

        self._sources['axis_factors']={p:list(sorted(set(list(self._sources['data'].data[str(p)])))) for p in self.categoricals}
        for p,f in self._sources['axis_factors'].items():
            if not len(f): f.append(('',)*len(p) if type(p) is not str else '')
    @pn.depends('_need_to_recreate_figure')
    def recreate_figures(self):
        self._figs:list[figure]=[]
        self._catranges:dict[str,FactorRange]={}
        for (px, py),fig_kwargs in zip(self.plot_pairs,self.fig_kwargs):
            for k,v in {'width':250,'height':300}.items(): fig_kwargs[k]=fig_kwargs.get(k,v)
            if px in self.categoricals:
                if False: # SEPARATE
                    self._catranges[px]=self._catranges.get(px,[])+[(fr:=FactorRange(*self._sources['axis_factors'][px]))]
                if True: # SHARE
                    if px not in self._catranges:
                        self._catranges[px]=[FactorRange(*self._sources['axis_factors'][px])]
                    fr=self._catranges[px][0]

                use_px=jitter(str(px),width=.4,range=fr)
                xrange_arg={"x_range":fr}
            else:
                use_px=str(px)
                xrange_arg={}
            self._figs.append((fig := figure(**fig_kwargs,**xrange_arg)))
            fig.circle(x=use_px,y=py,source=self._sources['data'],legend_field='legend',color='color')
            if px in self.stars and py in self.stars:
                fig.star(x=px,y=py,source=self._sources['stars'],size=15,fill_color='gold',line_color='black')
        for fig in self._figs: smaller_legend(fig)
        if self.fig_arrangement=='row':
            return bokeh.layouts.gridplot([self._figs],toolbar_location='right')
        elif self.fig_arrangement=='column':
            return bokeh.layouts.gridplot([[f] for f in self._figs],toolbar_location='right')

    def polish_figures(self):
        if (labels:=self._sources['labels']) is not None:
            for (px, py), fig in zip(self.plot_pairs,self._figs):
                fig.xaxis.axis_label=labels[px]
                fig.yaxis.axis_label=labels[py]
        for (px, py), fig in zip(self.plot_pairs,self._figs):
            if px in self.categoricals:
                for fr in self._catranges[px]:
                    fr.factors=self._sources['axis_factors'][px]


class WafermapFilterPlotter(FilterPlotter):

    fig_kwargs=hvparam.Parameter()
    plot_vars=hvparam.Parameter()
    plot_var_to_meas_group=hvparam.Parameter()
    shownames=hvparam.Parameter()
    _built_in_view_settings=['norm_by']
    wmap=hvparam.Parameter()

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    @property
    def _meas_group_to_plot_vars(self):
        if not hasattr(self,'__meas_group_to_plot_vars'):
            assert (self.meas_groups is not None)!=(self.plot_var_to_meas_group is not None), \
                "Supply *either* meas_groups *or* plot_var_to_meas_group"
            if self.meas_groups is not None:
                assert len(self.meas_groups)==1, \
                    "If using more than one meas_group for WafermapFilterPlotter, supply plot_var_to_meas_group instead"
                meas_groups=self.meas_groups
                plot_vars=[list(self.plot_vars)]
            if self.plot_var_to_meas_group is not None:
                meas_groups=sorted(self.plot_var_to_meas_group.values())
                plot_vars=[[p for p in self.plot_pairs
                            if self.plot_var_to_meas_group[p]==m] \
                           for m in meas_groups]
            self.__meas_group_to_plot_vars=dict(zip(meas_groups,plot_vars))
        return self.__meas_group_to_plot_vars

    @property
    def _plot_var_to_meas_group(self):
        if not hasattr(self,'__plot_var_to_meas_group'):
            assert (self.plot_vars is not None)!=(self.plot_var_to_meas_group is not None), \
                "Supply *either* plot_vars *or* plot_var_to_meas_group"
            if self.plot_vars is not None:
                self.__plot_var_to_meas_group={p:self.meas_groups[0] for p in self.plot_vars}
            if self.plot_var_to_meas_group is not None:
                self.__plot_var_to_meas_group=self.plot_var_to_meas_group
        return self.__plot_var_to_meas_group
    def get_meas_groups(self):
        return list(self._meas_group_to_plot_vars.keys())

    def get_scalar_column_names(self):
        filter_and_norm_cols=super().get_scalar_column_names()[0]
        return [(pv+filter_and_norm_cols) for pv in self._meas_group_to_plot_vars.values()]

    def get_raw_column_names(self):
        return [[] for m in self._meas_group_to_plot_vars]

    def update_sources(self,pre_sources):
        if len(self.get_meas_groups())!=1:
            # Haven't implemented merging multiple tables yet
            # Could still make use of current class if every plot pair is within a table, but haven't implemented that yet
            raise NotImplementedError

        dict_to_df=lambda dct: make_serializable(pd.DataFrame(dct).set_index('DieLB'))

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={m:ColumnDataSource({}) for m in self._meas_group_to_plot_vars}

            # Empty prototype
            for m,pp in self._meas_group_to_plot_vars.items():
                self._sources[m].data=Waferplot.make_source_dict_from_pre_source(dict_to_df(dict(**{p:[] for p in pp},**{'DieLB':[]})),self.wmap,fields=None)
            self._sources['labels']=None

        # Otherwise, analyze the real data
        else:

            # Compile it all to right columns
            for mi,(m,pp) in enumerate(self._meas_group_to_plot_vars.items()):
                self._sources[m].data=Waferplot.make_source_dict_from_pre_source(dict_to_df(dict(
                        **{
                            p:self._normalizer.get_scaled(pre_sources[mi],p,self.norm_by)
                            for p in pp
                        },**{'DieLB':pre_sources[mi]['DieLB']})),
                    self.wmap,fields=None)
            self._sources['labels']={}
            for p in self._plot_var_to_meas_group:
                eu=self._normalizer.formatted_endunits(p,self.norm_by)
                sh=self._normalizer.shorthand(p,self.norm_by)
                eupart=fr"\text{{ [{eu}]}}" if eu!="" else ""
                self._sources['labels'][p]=fr"$${self.shownames[p]}{sh}{eupart}$$"

    @pn.depends('_need_to_recreate_figure')
    def recreate_figures(self):
        self._figs={}
        for p,m in self._plot_var_to_meas_group.items():
            fig=self._figs[p]=WaferplotWithRangeControls(
                low=self.plot_vars[p].get('low',None),
                high=self.plot_vars[p].get('high',None),
                cmap=({'lin':LinearColorMapper,'log':LogColorMapper}[self.plot_vars[p].get('scale','lin')],
                      getattr(palettes,self.plot_vars[p].get('cmap','RdYlGn'))),
                color=p, die_lb=self.wmap,
                width=250,height=300,allow_tap=False,
                pre_transform=None, fig=None, pre_source=None, source=self._sources[m],colorbar=True)
            fig.fig.title.text='Wafermap'
        #self._figs=[]
        #for (px, py),fig_kwargs in zip(self.plot_pairs,self.fig_kwargs):
        #    self._figs.append((fig := figure(width=250,height=300,**fig_kwargs)))
        #    fig.circle(x=px,y=py,source=self._sources['data'],legend_field='legend',color='color')
        #    if px in self.stars and py in self.stars:
        #        fig.star(x=px,y=py,source=self._sources['stars'],size=15,fill_color='gold',line_color='black')
        #for fig in self._figs: smaller_legend(fig)
        return pn.Row(*self._figs.values())
        #return pn.VSpacer()

    def polish_figures(self):
        if (labels:=self._sources['labels']):
            for p,f in self._figs.items():
                f.fig.title.text=labels[p]
class WaferplotWithRangeControls(CompositeWidget):
    low  = hvparam.Number(allow_None=True)
    high = hvparam.Number(allow_None=True)
    lock = hvparam.Boolean()

    def __init__(self,source,low=0,high=1,lock=True,width=None,height=None,sizing_mode=None,*args,**kwargs):
        super().__init__(low=low,high=high,lock=lock,width=width,height=height,sizing_mode=sizing_mode)
        self._source=source
        cmap_kwargs={'low': self.low  if self.lock else None,
                     'high':self.high if self.lock else None,
                     'nan_color':'white'}
        self.fig=waferplot(*args, **kwargs, cmap_kwargs=cmap_kwargs,
                           source=source,
                           fig_kwargs=dict(width=width,sizing_mode='stretch_height'))

        fhi=pn.widgets.FloatInput.from_param(self.param.high,sizing_mode='stretch_width')
        flo=pn.widgets.FloatInput.from_param(self.param.low,sizing_mode='stretch_width')
        clk=pn.widgets.Checkbox.from_param(self.param.lock,name='',sizing_mode='stretch_width')
        c=pn.Column(self.fig,
                    pn.Row(flo, fhi,clk,sizing_mode='stretch_width'),
                    width=width,sizing_mode='stretch_height')

        self._composite[:]=[c]
        self.param.watch(self.range_param_changed,['low','high','lock'])

    def range_param_changed(self,event):
        trans=self.fig.select('cmap_transform')[0]
        with batch_call_watchers(self):
            if event.name=='high':
                self.lock=True
                trans.high=event.new
            elif event.name=='low':
                self.lock=True
                trans.low=event.new
            elif event.name=='lock':
                if event.new:
                    trans.low=self.low
                    trans.high=self.high
                else:
                    trans.low=None
                    trans.high=None

        # Hackish way to trigger re-rendering
        current_data=dict(self._source.data)
        #self._source.data={k:[] for k in current_data}
        self._source.data=current_data
