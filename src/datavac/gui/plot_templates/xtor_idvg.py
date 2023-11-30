import numpy as np
import bokeh.layouts
import panel as pn
from bokeh.models import ColumnDataSource

from bokeh_transform_utils.transforms import multi_abs_transform
import param as hvparam

from bokeh.plotting import figure

from datavac.gui.bokeh_util.util import make_color_col, smaller_legend
from datavac.gui.panel_util.filter_plotter import FilterPlotter
from datavac.util.logging import logger
from datavac.util.tables import stack_sweeps


class StandardIdVgPlotter(FilterPlotter):

    # Whether plotting nMOS or pMOS
    pol=hvparam.Selector(objects=['p','n'])

    # View settings
    sweep_dirs=hvparam.ListSelector(default=['f'],objects=['f','r'])
    vds=hvparam.ListSelector()
    _built_in_view_settings = ['norm_by','color_by','vds','sweep_dirs']

    def __init__(self,vds_options,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # Set options and defaults for the view settings
        self.param.color_by.objects=list(self.filter_settings.keys())+['VD','SweepDir']
        if self.color_by is None: self.color_by='LotWafer'
        self.param.vds.objects=vds_options
        if self.vds is None: self.vds=[next(iter(sorted(vds_options,key=lambda x: abs(float(x)),reverse=True)))]

    def get_raw_column_names(self):
        return [['VG']+[f'fI{term}@VD={vd}' for vd in self.param.vds.objects for term in ['G','D']]]

    def _extract_gm(self, stacked_data):
        if len(stacked_data[f'ID']):
            try:
                id=np.vstack(stacked_data[f'ID'])
                vg=np.vstack(stacked_data[f'VG'])
                stacked_data[f'GM']=list((np.gradient(id,axis=1).T/(vg[:,1]-vg[:,0])).T)
            except Exception as e:
                logger.error(f"Couldn't do GM extraction: {e}")
                logger.error(f"Usually this is because the data is not uniform.")
                stacked_data['GM']=list(id*np.NaN)
        else:
            stacked_data[f'GM']=[]

    def update_sources(self, pre_sources, event=None):

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'curves':ColumnDataSource({})}

            # Empty prototype
            self._sources['curves'].data={'VG':[],'ID':[],'IG':[],'GM':[], 'legend':[], 'color':[]}
            self._sources['ylabels']=None

        # Otherwise, analyze the real data
        else:

            # Stack the various columns (ie fID@VD=1 and fID@VD=2 get stacked to one column 'ID')
            idvg=stack_sweeps(pre_sources[0],'VG',['ID','IG'],'VD',
                              restrict_dirs=(self.sweep_dirs if self.sweep_dirs!=[] else None),
                              restrict_swv=(self.vds if  self.vds!=[] else None))
            # Add the GM column
            self._extract_gm(idvg)

            # Compile it all to right columns
            self._sources['curves'].data={
                'VG':idvg[f'VG'],
                'ID':self._normalizer.get_scaled(idvg,f'ID',self.norm_by),
                'IG':self._normalizer.get_scaled(idvg,f'IG',self.norm_by),
                'GM':self._normalizer.get_scaled(idvg,f'GM',self.norm_by),
                'legend':idvg[self.color_by],
                'color':make_color_col(idvg[self.color_by],
                           all_factors=self.param[self.color_by].objects)
            }

            # And make the y_axis names
            divstr=self._normalizer.shorthand('ID',self.norm_by)
            end_units_id=self._normalizer.formatted_endunits('ID',self.norm_by)
            end_units_gm=self._normalizer.formatted_endunits('GM',self.norm_by)
            self._sources['ylabels']={
                'idlog':fr'$$I_{{D,G}}{divstr}\text{{ [{end_units_id}]}}$$',
                'idlin':fr'$$I_{{D,G}}{divstr}\text{{ [{end_units_id}]}}$$',
                'gm':fr'$$G_{{M}}{divstr}\text{{ [{end_units_gm}]}}$$'
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
        figlog.multi_line(xs='VG',ys=multi_abs_transform('ID'),source=source,legend_field='legend',color='color')
        figlog.multi_line(xs='VG',ys=multi_abs_transform('IG'),source=source,line_dash='dashed',color='color')
        figlog.xaxis.axis_label="$$V_G\\text{ [V]}$$"

        figlin = self._figlin = figure(y_axis_type='linear',width=250,height=300,
                                       y_range=bokeh.models.DataRange1d(flipped=(self.pol=='p')))
        figlin.multi_line(xs='VG',ys='ID',source=source,legend_field='legend',color='color')
        figlin.multi_line(xs='VG',ys='IG',source=source,line_dash='dashed',color='color')
        figlin.xaxis.axis_label="$$V_G\\text{ [V]}$$"

        figgm = self._figgm = figure(y_axis_type='linear',width=250,height=300,)
        figgm.multi_line(xs='VG',ys='GM',source=source,legend_field='legend',color='color')
        figgm.xaxis.axis_label="$$V_G\\text{ [V]}$$"

        for fig in [figlog,figlin,figgm]: smaller_legend(fig)
        return bokeh.layouts.gridplot([[figlog,figlin,figgm]],toolbar_location='right')

    def polish_figures(self):
        if (ylabels:=self._sources['ylabels']) is not None:
            self._figlog.yaxis.axis_label=ylabels['idlog']
            self._figlin.yaxis.axis_label=ylabels['idlin']
            self._figgm.yaxis.axis_label=ylabels['gm']


