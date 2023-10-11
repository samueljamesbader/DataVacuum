from contextlib import contextmanager
import pandas as pd
import numpy as np

from bokeh.io import curdoc


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
