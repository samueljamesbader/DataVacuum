import pandas as pd
import numpy as np
import bokeh.layouts
from typing import Union, Optional, Callable, Any

import bokeh.core.properties
import bokeh.palettes
from bokeh.core.property.serialized import NotSerialized
from bokeh.model import DataModel
from bokeh.models import Transform, LinearColorMapper, CategoricalColorMapper, ColumnDataSource, CustomJS, ColorBar
from bokeh.models.widgets import Div
from pandas import DataFrame
from bokeh.plotting import figure
from functools import wraps
from bokeh.transform import linear_cmap, transform
from bokeh_transform_utils.transforms import compose_transforms, composite_transform


class ReloadableDataModel(DataModel):
    __implementation__=None


class Waferplot(ReloadableDataModel):

    selected_dielb=bokeh.core.properties.Any(default=None)
    _diemap=NotSerialized(bokeh.core.properties.Any(default=None))
    _color=NotSerialized(bokeh.core.properties.Any(default=None))
    fig=NotSerialized(bokeh.core.properties.Any(default=None))
    source=NotSerialized(bokeh.core.properties.Any(default=None))


    def __init__(self, color: str, die_lb: dict, cmap: Union[str,dict], fig_kwargs:dict={}, allow_tap:bool=True,
                 cmap_kwargs:dict = {}, pre_transform:Optional[Transform] = None, fig:Optional[figure]=None,
                 pre_source: Optional[DataFrame] = None, source: Optional[ColumnDataSource] = None, colorbar:bool=False):
        super().__init__()

        self._diemap=die_lb
        self._color=color

        if source is None:
            self.source=ColumnDataSource({})
            self.plot(pre_source)
        else:
            self.source=source

        if type(cmap)==tuple and type(cmap[0])!=str:
            transmap,cmap=cmap
        else:
            transmap=LinearColorMapper
        if type(cmap) in [tuple,str]:
            cmap_kwargs['low']=cmap_kwargs['low'] if 'low' in cmap_kwargs else np.nanmin([x for x in self.source.data[color] if x is not None])
            cmap_kwargs['high']=cmap_kwargs['high'] if 'high' in cmap_kwargs else np.nanmax([x for x in self.source.data[color] if x is not None])
            cmap_transform=transmap(palette=cmap,**cmap_kwargs,name='cmap_transform')
            cmapped_field=composite_transform(color,*[t for t in [pre_transform,cmap_transform] if t is not None])
            #cmapped_field=transmap(color,getattr(bokeh.palettes,cmap),**cmap_kwargs)
        if type(cmap)==list:
            assert not pre_transform
            cmap_kwargs['low']=cmap_kwargs['low'] if 'low' in cmap_kwargs else np.nanmin([x for x in self.source.data[color] if x is not None])
            cmap_kwargs['high']=cmap_kwargs['high'] if 'high' in cmap_kwargs else np.nanmax([x for x in self.source.data[color] if x is not None])
            cmap_transform=transmap(palette=cmap,**cmap_kwargs,name='cmap_transform')
            cmapped_field=transform(color,cmap_transform)
        elif type(cmap)==dict:
            categorical_transform=CategoricalColorMapper(
                palette=list(cmap.values()),factors=list(cmap.keys()),**cmap_kwargs,name='cmap_transform')
            cmapped_field=transform(color,compose_transforms(pre_transform,categorical_transform))

        TOOLTIPS = [
            ("", "@DieLB"),
        ]
        self.fig:figure = fig
        if self.fig is None:
            default_fig_kwargs=dict(width=100,height=100,toolbar_location=None,
                                    tooltips=TOOLTIPS,tools=("hover,tap" if allow_tap else 'hover'))
            default_fig_kwargs.update(fig_kwargs)
            self.fig=figure(**default_fig_kwargs,
                            x_range=[-self._diemap['diameter'] / 2, self._diemap['diameter'] / 2],
                            y_range=[-self._diemap['diameter'] / 2, self._diemap['diameter'] / 2])
        #self.fig.circle(x=0, y=0, radius=self._diemap['diameter'] / 2, line_color='black', fill_color=None, line_width=.25, hit_dilation=0)
        patches=self.fig.patches("x","y",source=self.source,line_color='black',line_width=.25,
                    fill_color=cmapped_field)
        self.fig.js_on_event('tap',CustomJS(args={'self':self},code="""
        console.log("In tap");
        console.log(self);
        console.log("Still in tap");
        if (self.source.selected.indices.length){
            self.selected_dielb=self.source.data['DieLB'][self.source.selected.indices[0]];
            self.source.selected.indices=[];
        }
        else {
            self.selected_dielb=null;
            self.source.selected.indices=[];
        }
        """))

        self.fig.xaxis.visible=False
        self.fig.xgrid.visible=False
        self.fig.yaxis.visible=False
        self.fig.ygrid.visible=False
        self.fig.outline_line_color=None

        if colorbar:
            color_bar = ColorBar(color_mapper=cmap_transform, width=10, label_standoff=2)
            self.fig.add_layout(color_bar,'right')

        self._inv_label=bokeh.models.Label(x=0,y=0,text='Duplicate Selection',angle=45,angle_units='deg',
                                           text_align='center',text_color='white',background_fill_color='black',
                                           visible=(len(self.source.data['DieLB'])!=len(set(self.source.data['DieLB']))))
        self.source.js_on_change('data',bokeh.models.CustomJS(args={'inv_label':self._inv_label},
                  code="""inv_label.visible=!((new Set(cb_obj.data['DieLB'])).size==cb_obj.data['DieLB'].length);"""))
        self.fig.add_layout(self._inv_label)

    @staticmethod
    def make_source_dict_from_pre_source(pre_source, diemap, fields):
        if pre_source is None:
            pre_source=diemap['patch_table'].copy()
            for field in fields:
                pre_source[field]=np.asarray([np.NaN] * len(pre_source), dtype='object')
            #pre_source["DieLB"]=''
            return ColumnDataSource.from_df(pre_source)
        else:
            return ColumnDataSource.from_df(
                pd.merge(left=diemap['patch_table'], right=pre_source,
                         how='left', left_index=True, right_index=True))

    def plot(self,pre_source):
        self.source.data=self.make_source_dict_from_pre_source(pre_source, self._diemap, [self._color])

@wraps(Waferplot.__init__)
def waferplot(*args,**kwargs):
    return Waferplot(*args,**kwargs).fig

class WaferplotGrid(ReloadableDataModel):

    selected_row_col_dielb=bokeh.core.properties.Any(default=None)
    times_clicked=bokeh.core.properties.Any(default=0)
    row_values=NotSerialized(bokeh.core.properties.Any(default=[None]))
    col_values=NotSerialized(bokeh.core.properties.Any(default=[None]))

    _cds_col_namer=NotSerialized(bokeh.core.properties.Any(default=None))
    _diemap=NotSerialized(bokeh.core.properties.Any(default=None))
    source=bokeh.core.properties.Any(default=None)
    lay=NotSerialized(bokeh.core.properties.Any(default=None))

    def __init__(self,
                 row_values: list[Any]=[None], row_labeler: Callable[[Any], str] = str,
                 col_values: list[Any]=[None], col_labeler: Callable[[Any], str] = str,
                 cds_col_namer=Callable[[Any, Any], str],
                 #row_col_die_callback:Optional[Callable[[Any,Any,str],None]]=None,
                 pre_source: Optional[DataFrame] = None, source: Optional[ColumnDataSource] = None, **waferplot_kwargs):
        super().__init__()

        self._diemap=waferplot_kwargs['die_lb']
        self.row_values=row_values
        self.col_values=col_values
        self._cds_col_namer=cds_col_namer

        if source is None:
            self.source=ColumnDataSource({})
            self.plot(pre_source)
        else:
            self.source=source

        lay=[]
        for rv in row_values:
            row:list[bokeh.models.LayoutDOM]=[]
            for cv in col_values:
                #if row_col_die_callback:
                #    die_callback=partial(row_col_die_callback,rv,cv)
                #    waferplot_kwargs.update(die_callback=die_callback)
                w=Waferplot(color=cds_col_namer(rv,cv),source=self.source,**waferplot_kwargs)
                row.append(w.fig)
                #fig.js_on_event('tap',CustomJS(args={'self':self},code="""
                #if (self.source.selected.indices.length)
                #    self.selected_dielb=self.source.data['DieLB'][self.source.selected.indices[0]];
                #else {
                #    self.selected_dielb=None;
                #    self.source.selected.indices=[];
                #}
                #"""))
                w.fig.js_on_event('tap',CustomJS(args={'self':self,'rv':rv,'cv':cv,'w':w},code="""
                    //alert("WPG");
                    self.selected_row_col_dielb=[rv,cv,w.selected_dielb];
                    //alert(rv);
                    //alert(cv);
                    //alert(cb_obj.selected_dielb);
                    self.times_clicked+=1;
                    //window.selfself=self;
                """))


            div=Div(text=row_labeler(rv),height=row[0].height,width=20)
            div.styles={'writing-mode':'vertical-rl','transform': 'rotate(180deg)',
                       'text-align':'center','height':'100%','text-decoration':'underline'}
            row=[bokeh.layouts.column(bokeh.layouts.Spacer(sizing_mode='stretch_height'),div,bokeh.layouts.Spacer(sizing_mode='stretch_height'))]+row
            lay+=[row]

        header_row=[None]
        for cv in col_values:
            div=Div(text=col_labeler(cv),width=row[1].width,height=10,sizing_mode='fixed',align='center')
            div.styles={'text-align':'center','text-decoration':'underline','margin':'0 auto','width':'100%'}
            #header_row.append(hcenter_with_spacers(div))
            header_row.append(div)
        #self.lay: bokeh.layouts.column=bokeh.layouts.column(lay)
        self.lay=bokeh.layouts.grid([header_row]+lay)

    def plot(self, pre_source):
        if pre_source is None:
            pre_source=self._diemap['patch_table'].copy()
            for rv in self.row_values:
                for cv in self.col_values:
                    pre_source[self._cds_col_namer(rv, cv)]=np.asarray([np.NaN] * len(pre_source), dtype='object')
            #pre_source["DieLB"]=''
            self.source.data= ColumnDataSource(pre_source).data.copy()
        else:
            self.source.data= ColumnDataSource(
                pd.merge(left=self._diemap['patch_table'],right=pre_source,
                         how='left',left_index=True,right_index=True)).data.copy()

@wraps(WaferplotGrid.__init__)
def waferplot_grid(*args,**kwargs):
    return WaferplotGrid(*args,**kwargs).lay