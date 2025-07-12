import numpy as np
import bokeh.layouts
import panel as pn
from bokeh.models import ColumnDataSource

from bokeh_transform_utils.transforms import multi_abs_transform
import param as hvparam

from bokeh.plotting import figure

from datavac.gui.bokeh_util.util import make_color_col, smaller_legend
from datavac.gui.panel_util.filter_plotter import FilterPlotter
from datavac.util.dvlogging import logger
from datavac.util.tables import stack_sweeps


class StandardIdVdPlotter(FilterPlotter):

    # Whether plotting nMOS or pMOS
    pol=hvparam.Selector(objects=['p','n'])

    # View settings
    sweep_dirs=hvparam.ListSelector(default=['f'],objects=['f','r'])
    vgs=hvparam.ListSelector()
    _built_in_view_settings = ['norm_by','color_by','vgs','sweep_dirs']

    def __init__(self,vgs_options,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # Set options and defaults for the view settings
        self.param.color_by.objects=list(self.filter_settings.keys())+['VG','SweepDir']
        if self.color_by is None: self.color_by='LotWafer'
        self.param.vgs.objects=vgs_options
        #if self.vgs is None: self.vgs=[next(iter(sorted(vgs_options,key=lambda x: abs(float(x)),reverse=True)))]

    def get_raw_column_names(self):
        return [['VD']+[f'{sd}I{term}@VG={vg}' for vg in self.param.vgs.objects for term in ['G','D'] for sd in self.param.sweep_dirs.objects]]

    def _extract_ro(self, stacked_data):
        if len(stacked_data[f'ID']):
            try:
                id=np.vstack(stacked_data[f'ID'])
                vd=np.vstack(stacked_data[f'VD'])
                stacked_data[f'RO']=list(1/(np.gradient(id,axis=1).T/(vd[:,1]-vd[:,0])).T)
            except Exception as e:
                logger.error(f"Couldn't do RO extraction: {e}")
                logger.error(f"Usually this is because the data is not uniform.")
                stacked_data['RO']=[id*np.nan for id in stacked_data['ID']]
        else:
            stacked_data[f'RO']=[]

    def update_sources(self, pre_sources, event=None):

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'curves':ColumnDataSource({})}

            # Empty prototype
            self._sources['curves'].data={'VD':[],'ID':[],'IG':[],'GM':[], 'legend':[], 'color':[]}
            self._sources['ylabels']=None

        # Otherwise, analyze the real data
        else:

            # Stack the various columns (ie fID@VD=1 and fID@VD=2 get stacked to one column 'ID')
            idvd=stack_sweeps(pre_sources[0],'VD',['ID','IG'],'VG',
                              restrict_dirs=(self.sweep_dirs if self.sweep_dirs!=[] else None),
                              restrict_swv=(self.vgs if  self.vgs!=[] else None))
            # Add the GM column
            self._extract_ro(idvd)

            # Compile it all to right columns
            self._sources['curves'].data={
                'VD':idvd[f'VD'],
                'ID':self._normalizer.get_scaled(idvd,f'ID',self.norm_by),
                'IG':self._normalizer.get_scaled(idvd,f'IG',self.norm_by),
                'RO':self._normalizer.get_scaled(idvd,f'RO',self.norm_by),
                'legend':idvd[self.color_by],
                'color':make_color_col(idvd[self.color_by],
                           all_factors=self.param[self.color_by].objects)
            }

            # And make the y_axis names
            divstr=self._normalizer.shorthand('ID',self.norm_by)
            end_units_id=self._normalizer.formatted_endunits('ID',self.norm_by)
            end_units_ro=self._normalizer.formatted_endunits('RO',self.norm_by)
            self._sources['ylabels']={
                'idlog':fr'$$I_{{D,G}}{divstr}\text{{ [{end_units_id}]}}$$',
                'idlin':fr'$$I_{{D,G}}{divstr}\text{{ [{end_units_id}]}}$$',
                'ro':fr'$$R_{{O}}{divstr}\text{{ [{end_units_ro}]}}$$'
            }

        #print("Updated sources")
        #import pdb; pdb.set_trace()
        #print(self._sources['curves'].data)
        #print(self._sources['ylabels'])


    @pn.depends('_need_to_recreate_figure')
    def recreate_figures(self):
        logger.debug("Creating figure")
        source=self._sources['curves']

        figlog = self._figlog = figure(y_axis_type='log',width=250,height=300)
        figlog.multi_line(xs='VD',ys=multi_abs_transform('ID'),source=source,legend_field='legend',color='color')
        figlog.multi_line(xs='VD',ys=multi_abs_transform('IG'),source=source,line_dash='dashed',color='color')
        figlog.xaxis.axis_label="$$V_D\\text{ [V]}$$"

        figlin = self._figlin = figure(y_axis_type='linear',width=250,height=300,
                                       y_range=bokeh.models.DataRange1d(flipped=(self.pol=='p')))
        figlin.multi_line(xs='VD',ys='ID',source=source,legend_field='legend',color='color')
        figlin.multi_line(xs='VD',ys='IG',source=source,line_dash='dashed',color='color')
        figlin.xaxis.axis_label="$$V_D\\text{ [V]}$$"

        figro = self._figro = figure(y_axis_type='linear',width=250,height=300,)
        figro.multi_line(xs='VD',ys='RO',source=source,legend_field='legend',color='color')
        figro.xaxis.axis_label="$$V_D\\text{ [V]}$$"

        for fig in [figlog,figlin,figro]: smaller_legend(fig)
        return bokeh.layouts.gridplot([[figlog,figlin,figro]],toolbar_location='right')

    def polish_figures(self):
        if (ylabels:=self._sources['ylabels']) is not None:
            self._figlog.yaxis.axis_label=ylabels['idlog']
            self._figlin.yaxis.axis_label=ylabels['idlin']
            self._figro.yaxis.axis_label=ylabels['ro']


