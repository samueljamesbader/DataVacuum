import logging

import numpy as np
import bokeh.layouts
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, CustomJSFilter

from bokeh_transform_utils.transforms import MultiAbsTransform, multi_abs_transform
from datavac.examples.filter_plotter_layout import AllAroundFilterPlotter
import param as hvparam

from bokeh.plotting import figure

from datavac import units
from datavac.gui.bokeh_util.palettes import get_sam_palette
from datavac.logging import logger

#def extract_gm(self,data):
#    for vd in self.vds:
#        if len(data[f'fID@VD={vd}']):
#            id=np.vstack(data[f'fID@VD={vd}'])
#            vg=np.vstack(data[f'VG'])
#            data[f'fGM@VD={vd}']=list((np.gradient(id,axis=1).T/(vg[:,1]-vg[:,0])).T)
#        else:
#            data[f'fGM@VD={vd}']=[]
def smaller_legend(fig):
    fig.legend.margin=0
    fig.legend.spacing=0
    fig.legend.padding=4
    fig.legend.label_text_font_size='8pt'
    fig.legend.label_height=10
    fig.legend.label_text_line_height=10
    fig.legend.glyph_height=10

def get_norm_scale(start_units, normalizer, end_units):
    #start=units.parse_units(start.split("[")[-1].split("]")[0] if "[" in start else "1")
    start=units.parse_units(start_units)
    normalizer=units.parse_units(normalizer.split("[")[-1].split("]")[0] if "[" in normalizer else "1")
    end=units.parse_units(end_units)
    scale=(1*start/normalizer).to(end).magnitude
    return scale

def stack_sweeps(df,x,ys,swv, restrict_dirs=None, restrict_swv=None):
    restrict_dirs=restrict_dirs if restrict_dirs else ('f','r')
    potential_starts=[f"{d}{y}@{swv}" for y in ys for d in restrict_dirs]
    yheaders=[k for k in df.columns if k.split("=")[0] in potential_starts]
    vals=list(set([yheader.split("=")[1] for yheader in yheaders]))
    if restrict_swv: vals=[v for v in vals if v in restrict_swv]
    bystanders=[c for c in df.columns if c not in yheaders and c!=x]
    subtabs=[]
    for v in vals:
        for d in restrict_dirs:
            yheaders_in_subtab=[yheader for yheader in yheaders
                                if yheader.startswith(d) and yheader.endswith(f"@{swv}={v}")]
            if not len(yheaders_in_subtab): continue
            subtabs.append(
                df[[x]+yheaders_in_subtab+bystanders]\
                    .rename(columns={yheader:yheader[1:].split("@")[0]\
                                     for yheader in yheaders_in_subtab})\
                    .assign(**{swv:v,'SweepDir':d}))
    if len(subtabs):
        return pd.concat(subtabs)
    else:
        return pd.DataFrame({k:[] for k in [x]+ys+bystanders})


class StandardIdVgPlotter(AllAroundFilterPlotter):
    normalizations=hvparam.Parameter({}, instantiate=True)
    pol=hvparam.Selector(objects=['p','n'])

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.param.color_by.objects=list(self.filter_settings.keys())+['VD','SweepDir']
        if self.color_by is None: self.color_by='LotWafer'

    def get_raw_column_names(self):
        return [['VG']+[f'fI{term}@VD={vd}' for vd in self.param.vds.objects for term in ['G','D']]]

    def get_scalar_column_names(self):
        return [list(self.normalizations.keys())+list(self.filter_settings.keys())]

    def extract_gm(self,stacked_data):
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

    def update_sources(self, pre_sources):
        if pre_sources is None:
            # Be sure to never remake a ColumnDataSource!
            if self._sources is None:
                self._sources={'curves':ColumnDataSource({}),'ylabels':None}

            # Empty prototype
            self._sources['curves'].data={'VG':[],'ID':[],'IG':[],'GM':[], 'legend':[], 'color':[]}
        else:
            directions=self.sweep_dirs if 'sweep_dirs' in self.param and self.sweep_dirs!=[] else None
            vds=self.vds if 'vds' in self.param and self.vds!=[] else None
            idvg=stack_sweeps(pre_sources[0],'VG',['ID','IG'],'VD',restrict_dirs=directions,restrict_swv=vds)

            self.extract_gm(idvg)

            factors=list(sorted(idvg[self.color_by].unique()))
            palette=get_sam_palette(len(factors))
            color_col=idvg[self.color_by].map(dict(zip(factors,palette))).astype('string')

            norm_by=self.norm_by if ('norm_by' in self.param and self.norm_by!='None') else '1'
            norm_deets=self.normalizations.get(norm_by,{'endunits':'A','shorthand':''})
            end_units=norm_deets['endunits']
            divstr=("/"+norm_deets['shorthand']) if norm_deets['shorthand']!='' else ''
            scale=get_norm_scale('A',norm_by,end_units=end_units)
            norm=(idvg[self.norm_by] if ('norm_by' in self.param and self.norm_by!='None') else 1)/scale
            self._sources['curves'].data={
                'VG':idvg[f'VG'],
                'ID':idvg[f'ID']/norm,
                'IG':idvg[f'IG']/norm,
                'GM':idvg[f'GM']/norm,
                'legend':idvg[self.color_by],
                'color':color_col
            }
            self._sources['ylabels']={
                'idlog':fr'$$I_{{D,G}}{divstr}\text{{ [{end_units}]}}$$',
                'idlin':fr'$$I_{{D,G}}{divstr}\text{{ [{end_units}]}}$$',
                'gm':fr'$$G_{{M}}{divstr}\text{{ [{end_units.replace("A","S")}]}}$$'}

        #print("Updated sources")
        #print(self._sources['curves'].data)


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


