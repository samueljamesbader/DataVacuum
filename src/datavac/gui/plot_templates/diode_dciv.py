import numpy as np
import bokeh.layouts
import panel as pn
from bokeh.models import ColumnDataSource

from bokeh_transform_utils.transforms import multi_abs_transform
from datavac.examples.filter_plotter_layout import AllAroundFilterPlotter
import param as hvparam

from bokeh.plotting import figure

from datavac.gui.bokeh_util.util import make_color_col, smaller_legend
from datavac.logging import logger


class StandardDiodeDCPlotter(AllAroundFilterPlotter):

    # View settings
    sweep_dirs=hvparam.ListSelector(default=['f'],objects=['f'])
    _built_in_view_settings = ['norm_by','color_by']

    def get_raw_column_names(self):
        return [['V','I']]

    def update_sources(self, pre_sources, event=None):

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'curves':ColumnDataSource({})}

            # Empty prototype
            self._sources['curves'].data={'V':[], 'I':[], 'legend':[], 'color':[]}
            self._sources['ylabels']=None

        # Otherwise, analyze the real data
        else:

            iv=pre_sources[0]

            # Compile it all to right columns
            self._sources['curves'].data={
                'V':iv[f'V'],
                'I':self._normalizer.get_scaled(iv,f'I',self.norm_by),
                'legend':iv[self.color_by],
                'color':make_color_col(iv[self.color_by],
                           all_factors=self.param[self.color_by].objects)
            }

            # And make the y_axis names
            divstr=self._normalizer.shorthand('I',self.norm_by)
            end_units_i=self._normalizer.formatted_endunits('I',self.norm_by)
            self._sources['ylabels']={
                'ilog':fr'$$I{divstr}\text{{ [{end_units_i}]}}$$',
                'ilin':fr'$$I{divstr}\text{{ [{end_units_i}]}}$$',
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
        figlog.multi_line(xs='V',ys=multi_abs_transform('I'),source=source,legend_field='legend',color='color')
        figlog.multi_line(xs='V',ys=multi_abs_transform('I'),source=source,line_dash='dashed',color='color')
        figlog.xaxis.axis_label="$$V\\text{ [V]}$$"

        figlin = self._figlin = figure(y_axis_type='linear',width=250,height=300)
        figlin.multi_line(xs='V',ys='I',source=source,legend_field='legend',color='color')
        figlin.multi_line(xs='V',ys='I',source=source,line_dash='dashed',color='color')
        figlin.xaxis.axis_label="$$V\\text{ [V]}$$"

        for fig in [figlog,figlin]: smaller_legend(fig)
        return bokeh.layouts.gridplot([[figlog,figlin]],toolbar_location='right')

    def polish_figures(self):
        if (ylabels:=self._sources['ylabels']) is not None:
            self._figlog.yaxis.axis_label=ylabels['ilog']
            self._figlin.yaxis.axis_label=ylabels['ilin']


