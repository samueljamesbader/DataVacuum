import numpy as np
import bokeh.layouts
import panel as pn
from bokeh.models import ColumnDataSource

import param as hvparam

from bokeh.plotting import figure

from datavac.gui.bokeh_util.util import make_color_col, smaller_legend
from datavac.gui.panel_util.filter_plotter import FilterPlotter
from datavac.util.logging import logger
from datavac.util.tables import stack_sweeps


class StandardCVPlotter(FilterPlotter):

    # View settings
    sweep_dirs=hvparam.ListSelector(default=['f'],objects=['f','r'])
    shift_by=hvparam.Selector()
    freqs=hvparam.ListSelector()
    _freqstrs_to_freqfloats={}
    _built_in_view_settings = ['norm_by','shift_by','color_by','freqs','sweep_dirs']

    def __init__(self,freqs_options,shift_by_options,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # Set options and defaults for the view settings
        self.param.color_by.objects=list(self.filter_settings.keys())+['Freq','SweepDir']
        if self.color_by is None: self.color_by='LotWafer'
        assert all([f.endswith('k') for f in freqs_options])
        self._freqstrs_to_freqfloats={f:float(f[:-1])*1e3 for f in freqs_options}
        self.param.freqs.objects=freqs_options
        if self.freqs is None:
            self.freqs=freqs_options
        self.param.shift_by.objects=shift_by_options
        if self.shift_by is None: self.shift_by=self.param.shift_by.objects[0]

    def get_raw_column_names(self):
        return [['VG']+[f'{sd}{comp}@freq={freq}' for freq in self.param.freqs.objects for comp in ['Cp','G'] for sd in self.param.sweep_dirs.objects]]

    def get_scalar_column_names(self):
        return [super().get_scalar_column_names()[0]+[k for k in self.param.shift_by.objects if k!='None']]

    def update_sources(self, pre_sources, event=None):

        # If no data to work with yet, make an empty prototype for figure creation
        if pre_sources is None:

            # Wrap this in a check for pre-existing self._sources
            # to ensure we never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'curves':ColumnDataSource({})}

            # Empty prototype
            self._sources['curves'].data={'VG':[],'Cp':[],'theta':[], 'legend':[], 'color':[]}
            self._sources['ylabels']=None

        # Otherwise, analyze the real data
        else:

            # Stack the various columns (ie fCp@freq=100k and fCp@freq=200k get stacked to one column 'CP')
            cv=stack_sweeps(pre_sources[0],'VG',['Cp','G'],'freq',
                              restrict_dirs=(self.sweep_dirs if self.sweep_dirs!=[] else None),
                              restrict_swv=(self.freqs if  self.freqs!=[] else None))

            cv['MeasLength']=[len(a) for a in cv['Cp']]
            cv=cv[cv['MeasLength']>0]

            f=cv['freq'].map(self._freqstrs_to_freqfloats)
            if len(cv):
                skip_theta=any(list(cv['MeasLength'].to_numpy()!=cv['MeasLength'].iloc[0]))
                if skip_theta:
                    theta=cv['VG']*np.NaN
                else:
                    theta=(180/np.pi)*np.arctan2(2*np.pi*f.to_numpy()*np.vstack(cv['Cp']-cv['Copen [F]']).T,np.vstack(cv['G']).T).T
            else:
                theta=[]

            if self.shift_by!='None':
                cv['Cp']-=cv[self.shift_by]

            # Compile it all to right columns
            self._sources['curves'].data={
                'VG':cv[f'VG'],
                'Cp':self._normalizer.get_scaled(cv,f'Cp',self.norm_by),
                'theta': list(theta),
                'legend':cv[{'Freq':'freq'}.get(self.color_by,self.color_by)],
                'color':make_color_col(cv[{'Freq':'freq'}.get(self.color_by,self.color_by)],
                           all_factors=self.param[{'Freq':'freqs'}.get(self.color_by,self.color_by)].objects)
            }

            # And make the y_axis names
            divstr=self._normalizer.shorthand('ID',self.norm_by)
            end_units_c=self._normalizer.formatted_endunits('Cp',self.norm_by)
            self._sources['ylabels']={
                'C':fr'$$C{divstr}\text{{ [{end_units_c}]}}$$',
                'G':fr'$$\text{{Phase angle}}\text{{ [$^\circ$]}}$$'
            }

        #print("Updated sources")
        #import pdb; pdb.set_trace()
        #print(self._sources['curves'].data)
        #print(self._sources['ylabels'])


    @pn.depends('_need_to_recreate_figure')
    def recreate_figures(self):
        logger.debug("Creating figure")
        source=self._sources['curves']

        figC = self._figC = figure(width=250,height=300)
        figC.multi_line(xs='VG',ys='Cp',source=source,legend_field='legend',color='color')
        figC.xaxis.axis_label="$$"+self.shownames.get('VG','V_G\\text{ [V]}')+"$$"

        figG = self._figG = figure(width=250,height=300,y_range=(-180,180))
        figG.multi_line(xs='VG',ys='theta',source=source,legend_field='legend',color='color')
        figG.xaxis.axis_label="$$"+self.shownames.get('VG','V_G\\text{ [V]}')+"$$"
        figG.add_layout((bokeh.models.Span(location=100,dimension='width',line_color='black',line_width=.5,line_dash='dashed')))
        figG.add_layout((bokeh.models.Span(location= 80,dimension='width',line_color='black',line_width=.5,line_dash='dashed')))
        figG.legend.location='bottom_center'

        for fig in [figC,figG]: smaller_legend(fig)
        return bokeh.layouts.gridplot([[figC,figG]],toolbar_location='right')

    def polish_figures(self):
        if (ylabels:=self._sources['ylabels']) is not None:
            print("Setting ylabels")
            self._figC.yaxis.axis_label=ylabels['C']
            self._figG.yaxis.axis_label=ylabels['G']


