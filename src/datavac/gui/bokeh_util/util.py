from contextlib import contextmanager
import pandas as pd
import numpy as np

from bokeh.io import curdoc

from datavac.gui.bokeh_util.palettes import get_sam_palette


@contextmanager
def hold_bokeh():
    curdoc().hold()
    try:
        yield
    finally:
        curdoc().unhold()

def make_serializable(df):
    df=df.copy()
    for k in df.columns:
        if str(df[k].dtype) in ['Float64','Float32']:
            df[k]=df[k].astype('float32')
        if str(df[k].dtype) in ['boolean']:
            df[k]=df[k].astype('object')
            df[k]=df[k].where(~pd.isna(df[k]),np.NaN)
    return df

def smaller_legend(fig):
    fig.legend.margin=0
    fig.legend.spacing=0
    fig.legend.padding=4
    fig.legend.label_text_font_size='8pt'
    fig.legend.label_height=10
    fig.legend.label_text_line_height=10
    fig.legend.glyph_height=10

def make_color_col(factor_col):
    factors=list(sorted(factor_col.unique()))
    return factor_col \
        .map(dict(zip(factors,get_sam_palette(len(factors))))) \
        .astype('string')
