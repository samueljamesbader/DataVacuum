import bokeh.layouts
import panel as pn
from bokeh.models import ColumnDataSource

import param as hvparam

from bokeh.plotting import figure

from datavac.gui.bokeh_util.util import make_color_col, smaller_legend
from datavac.gui.panel_util.filter_plotter import FilterPlotter
from datavac.util.dvlogging import logger
from datavac.util.units import Normalizer


class StandardTLMIVPlotter(FilterPlotter):

    # View settings
    sweep_dirs=hvparam.ListSelector(default=['f'],objects=['f'])
    _built_in_view_settings = ['norm_by','color_by']
    r_name=hvparam.String(default='R')
    lsep_name=hvparam.String()

    def get_raw_column_names(self):
        #return [['V','I'],False] # two meas groups
        return [['V','I']]
    def get_scalar_column_names(self):
        su=super().get_scalar_column_names()
        norm_by=([self.norm_by] if self.norm_by is not 'None' else [])
        #return [[self.r_name,self.lsep_name]+su[0],[*norm_by,self.color_by,'Rdrift [ohm/um]','Rc [ohm]']] # two meas groups
        return [[self.r_name,self.lsep_name]+su[0]]

    def update_sources(self, pre_sources, event=None):

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'curves':ColumnDataSource({})}

            # Empty prototype
            self._sources['curves'].data={'V':[], 'I':[], 'R':[], 'L':[], 'legend':[], 'color':[]}
            self._sources['ylabels']=None

        # Otherwise, analyze the real data
        else:

            iv=pre_sources[0]

            # Compile it all to right columns
            self._sources['curves'].data={
                'V':iv[f'V'],
                'I':self._normalizer.get_scaled(iv,f'I',self.norm_by),
                'R':self._normalizer.get_scaled(iv,self.r_name,self.norm_by),
                'L':self._normalizer.get_scaled(iv,self.lsep_name,self.norm_by),
                'legend':iv[self.color_by],
                'color':make_color_col(iv[self.color_by],
                           all_factors=self.param[self.color_by].objects)
            }

            # And make the y_axis names
            divstr=self._normalizer.shorthand('I',self.norm_by)
            end_units_i=self._normalizer.formatted_endunits('I',self.norm_by)
            end_units_r=self._normalizer.formatted_endunits(self.r_name,self.norm_by)
            end_units_l=self._normalizer.formatted_endunits(self.lsep_name,self.norm_by)
            #end_units_l=self._normalizer.formatted_endunits(self.lsep_name,self.norm_by)
            self._sources['ylabels']={
                'ilin':fr'$$I{divstr}\text{{ [{end_units_i}]}}$$',
                'r':fr'$$R{self._normalizer.shorthand(self.r_name,self.norm_by)}\text{{ [{end_units_r}]}}$$',
            }

        #print("Updated sources")
        #import pdb; pdb.set_trace()
        #print(self._sources['curves'].data)
        #print(self._sources['ylabels'])


    @pn.depends('_need_to_recreate_figure')
    def recreate_figures(self):
        logger.debug("Creating figure")
        source=self._sources['curves']

        figlin = self._figlin = figure(y_axis_type='linear',width=350,height=300)
        figlin.multi_line(xs='V',ys='I',source=source,legend_field='legend',color='color')
        figlin.multi_line(xs='V',ys='I',source=source,line_dash='dashed',color='color')
        figlin.xaxis.axis_label="$$V\\text{ [V]}$$"

        figrvs = self._figrvs = figure(y_axis_type='linear',width=350,height=300)
        figrvs.circle(x='L',y='R',source=source,legend_field='legend',color='color')
        print("X name")
        print(Normalizer.clean_up_column_name_for_label(self.lsep_name))
        figrvs.xaxis.axis_label=Normalizer.clean_up_column_name_for_label(self.lsep_name)
        figrvs.legend.location='top_left'

        for fig in [figlin,figrvs]: smaller_legend(fig)
        return bokeh.layouts.gridplot([[figlin,figrvs]],toolbar_location='right')

    def polish_figures(self):
        if (ylabels:=self._sources['ylabels']) is not None:
            self._figlin.yaxis.axis_label=ylabels['ilin']
            self._figrvs.yaxis.axis_label=ylabels['r']


