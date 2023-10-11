import numpy as np
import bokeh.layouts
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource

from bokeh_transform_utils.transforms import MultiAbsTransform, multi_abs_transform
from datavac.examples.filter_plotter_layout import AllAroundFilterPlotter
import param as hvparam

from bokeh.plotting import figure

from datavac import units
def get_norm_scale(start_units, normalizer, end_units):
    #start=units.parse_units(start.split("[")[-1].split("]")[0] if "[" in start else "1")
    start=units.parse_units(start_units)
    normalizer=units.parse_units(normalizer.split("[")[-1].split("]")[0] if "[" in normalizer else "1")
    end=units.parse_units(end_units)
    scale=(1*start/normalizer).to(end).magnitude
    return scale

class StandardIdVgPlotter(AllAroundFilterPlotter):
    normalizations=hvparam.Parameter({}, instantiate=True)
    pol=hvparam.Selector(objects=['p','n'])

    def __init__(self,*args,**kwargs):
        self.param.add_parameter('_y1_label',hvparam.String("$$I_{D,G}$$"))
        self.param.add_parameter('_y2_label',hvparam.String("$$I_{D,G}$$"))
        self.param.add_parameter('_y3_label',hvparam.String("$$G_M$$"))
        super().__init__(*args,**kwargs)
    def get_raw_column_names(self):
        return [['VG']+[f'fI{term}@VD={vd}' for vd in self.param.vds.objects for term in ['G','D']]]
    def get_scalar_column_names(self):
        return [list(self.normalizations.keys())+list(self.filter_settings.keys())]

    def extract_gm(self,data):
        for vd in self.vds:
            if len(data[f'fID@VD={vd}']):
                id=np.vstack(data[f'fID@VD={vd}'])
                vg=np.vstack(data[f'VG'])
                data[f'fGM@VD={vd}']=list((np.gradient(id,axis=1).T/(vg[:,1]-vg[:,0])).T)
            else:
                data[f'fGM@VD={vd}']=[]

    def update_sources(self, pre_sources):
        if pre_sources is None:
            self._sources={
                'curves':ColumnDataSource({'VG':[],'ID':[],'IG':[],'GM':[], 'color':[]})
            }
        else:
            idvg=pre_sources[0]
            self.extract_gm(idvg)

            if 'color_by' not in self.param:
                color=['blue']*(len(idvg)*len(self.vds))
            else:
                if self.param.color_by=='vds':
                    color=
            color_by=self.color_by if ('color_by' in self.param)

            norm_by=self.norm_by if ('norm_by' in self.param and self.norm_by!='None') else '1'
            norm_deets=self.normalizations.get(norm_by,{'endunits':'A','shorthand':''})
            end_units=norm_deets['endunits']
            divstr=("/"+norm_deets['shorthand']) if norm_deets['shorthand']!='' else ''
            scale=get_norm_scale('A',norm_by,end_units=end_units)
            norm=(idvg[self.norm_by] if ('norm_by' in self.param and self.norm_by!='None') else 1)/scale
            self._sources['curves'].data={
                'VG':list(pd.concat([idvg[f'VG'] for vd in self.vds])),
                'ID':list(pd.concat([idvg[f'fID@VD={vd}']/norm for vd in self.vds])),
                'IG':list(pd.concat([idvg[f'fIG@VD={vd}']/norm for vd in self.vds])),
                'GM':list(pd.concat([idvg[f'fGM@VD={vd}']/norm for vd in self.vds]))
            }
            self._y1_label=fr'$$I_{{D,G}}{divstr}\text{{ [{end_units}]}}$$'
            self._y2_label=fr'$$I_{{D,G}}{divstr}\text{{ [{end_units}]}}$$'
            self._y3_label=fr'$$G_{{M}}{divstr}\text{{ [{end_units.replace("A","S")}]}}$$'


    @pn.depends('_need_to_recreate_figure')
    def recreate_figures(self):
        print("RECREATING FIGURE")
        source=self._sources['curves']

        figlog = self._figlog = figure(y_axis_type='log',width=250,height=300)
        figlog.multi_line(xs='VG',ys=multi_abs_transform('ID'),source=source)
        figlog.multi_line(xs='VG',ys=multi_abs_transform('IG'),source=source,line_dash='dashed')
        figlog.xaxis.axis_label="$$V_G\\text{ [V]}$$"

        figlin = self._figlin = figure(y_axis_type='linear',width=250,height=300,
                                       y_range=bokeh.models.DataRange1d(flipped=(self.pol=='p')))#start=0,end=np.min((source.data['ID'])),
                                                                        #bounds=(None,0) if self.pol=='p' else (0,None)))
        #if self.pol=='p': figlin.y_range.start=0
        figlin.multi_line(xs='VG',ys='ID',source=source)
        figlin.multi_line(xs='VG',ys='IG',source=source,line_dash='dashed')
        figlin.xaxis.axis_label="$$V_G\\text{ [V]}$$"

        figgm = self._figgm = figure(y_axis_type='linear',width=250,height=300,)
                                     #y_range=bokeh.models.DataRange1d(bounds=(0,None),start=0))
        figgm.multi_line(xs='VG',ys='GM',source=source)
        figgm.xaxis.axis_label="$$V_G\\text{ [V]}$$"
        return bokeh.layouts.gridplot([[figlog,figlin,figgm]],toolbar_location='right')

    def polish_figures(self):
        self._figlin.yaxis.axis_label=self._y1_label
        self._figlog.yaxis.axis_label=self._y2_label
        self._figgm.yaxis.axis_label=self._y3_label
