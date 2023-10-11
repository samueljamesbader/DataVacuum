import re
from collections import deque
from time import perf_counter
from contextlib import contextmanager
import pandas as pd
from pandas import DataFrame
from datavac.logging import logger

def last(it):
    return deque(it,maxlen=1).pop()
def first(it):
    return next(iter(it))
def only(seq,message=None):
    assert len(seq)==1, (message if message else f"This list should have exactly one element: {seq}.")
    return seq[0]
def only_row(df,message=None):
    assert len(df)==1, (message if message else f"This table should have only one row {str(df)}")
    return df.iloc[0]

@contextmanager
def count_perf(message):
    start=perf_counter()
    yield
    end=perf_counter()
    logger.debug(f"{message}: {end-start:0.2f}s")

def wide_notebook():
    from IPython.display import display, HTML
    display(HTML("<style>.container { width:85% !important; }</style>"))

def drop_multilevel_columns(df: DataFrame) -> DataFrame:
    """ Returns a DataFrame containing only the columns that are "effectively single-level".

    eg if a DataFrame has multi-index columns [('A',''),('B',''),('C',1),('C',2)], this returns
    a DataFrame with columns ['A','B].  If a DataFrame is already not a multi-index, it will be returned as-is.
    """
    if type(df.columns) is pd.core.indexes.multi.MultiIndex:
        df=df[[col for col in df.columns if all(c=='' for c in col[1:])]]
        df.columns=[''.join(col) for col in df.columns]
        return df
    else:
        return df

def stack_keeping_bystanders(df: DataFrame, level: int) -> DataFrame:
    assert type(df.columns) is pd.core.indexes.multi.MultiIndex
    level=level if level >= 0 else len(next(iter(df.columns))) + level
    the_index=df.index.names
    df=df.reset_index(drop=False)
    #import pdb; pdb.set_trace()
    assert level!=0
    bystanders=[col for col in df.columns if all(c=='' for c in col[1:])]
    #bystanders=[col for col in df.columns if col[level] =='']
    df=df.set_index([b[0] for b in bystanders])
    df=df.stack(level)
    df=df.reset_index(drop=False)
    df=df.set_index(the_index)
    return df


def flatten_multilevel_columns(df: DataFrame, separator: str = ' ') -> DataFrame:
    """ Returns a DataFrame in which multi-index columns have been flattened to a single-index.

    eg if a DataFrame has multi-index columns [('A',''), ('B',''),('C',1),('C',2)] this returns
    a DataFrame with columns ['A', 'B', 'C 1', 'C 2'].

    The separator can be changed from its default of space (' ').
    """
    if type(df.columns) is pd.core.indexes.multi.MultiIndex:
        df.columns=[' '.join([str(x) for x in col if x!='']) for col in df.columns]
        return df
    else:
        return df

def check_dtypes(dataframe:DataFrame):
    assert 'in_place' not in dataframe.columns, "There's a column named 'in_place', this is probably a mistake..."
    for c,dtype in dataframe.dtypes.items():
        assert str(dtype)!='object', \
            f"Column '{c}' has dtype object! Usually this is supposed to be string or some nullable type"
    return dataframe
