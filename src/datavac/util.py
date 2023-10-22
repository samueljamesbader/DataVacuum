import re
import numpy as np
from collections import deque
from time import perf_counter
from contextlib import contextmanager
import pandas as pd
from pandas import DataFrame

from datavac import units
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

class Normalizer():
    def __init__(self, deets):
        self._udeets={}
        self._shorthands={}
        for n,(shorthand,ninfo) in deets.items():
            self._shorthands[n]=shorthand
            self._udeets[n]={}
            if '[' in n:
                norm_units=units.parse_expression(n.split('[')[1].split(']')[0])
            else:
                norm_units=units.parse_units('1')
            for ks,kinfo in ninfo.items():
                if type(ks) is not tuple:
                    ks=(ks,)
                for k in ks:
                    self._udeets[n][k]=kinfo.copy()
                    if '[' in k:
                        start_units_from_name=k.split('[')[1].split(']')[0]
                        if ('start_units' in self._udeets[n][k]) \
                                and self._udeets[n][k]['start_units']!=start_units_from_name:
                            raise Exception(f"Under '{n}', column '{k}'" \
                                            f"has conflicting start units {self._udeets[n][k]['start_units']}")
                        else:
                            self._udeets[n][k]['start_units']=start_units_from_name
                    else:
                        if 'start_units' not in self._udeets[n][k]:
                            raise Exception(f"Under '{n}', no start_units provided or read from column '{k}'")
                    start_units=units.parse_expression(self._udeets[n][k]['start_units'])
                    end_units=units.parse_units(self._udeets[n][k]['end_units'])
                    assert (ntype:=self._udeets[n][k]['type']) in ['*', '/']
                    nstart_units=start_units/norm_units if ntype=='/' else start_units*norm_units
                    try:
                        self._udeets[n][k]['units_scale_factor']=nstart_units.to(end_units).magnitude
                    except Exception as e:
                        logger.error(f"Couldn't convert '{k}' in {start_units} with normalization '{n}' to {end_units}")
                        raise e

    def get_scaled(self, df, column, normalizer):
        if column not in self._udeets[normalizer]:
            logger.debug(f"Normalizer: {normalizer} does not interact with {column}")
            return df[column]
        ntype=self._udeets[normalizer][column]['type']
        scale=self._udeets[normalizer][column]['units_scale_factor']
        column=df[column]
        normalizer=df[normalizer] if normalizer!='None' else 1
        return (column/normalizer if ntype=='/' else column*normalizer)*scale

    def shorthand(self, column, normalizer):
        if column not in self._udeets[normalizer]:
            #logger.debug(f"Normalizer: {normalizer} does not interact with {column}")
            return ""
        t={'/':'/','*':r'\cdot '}[self._udeets[normalizer][column]['type']]
        sh=self._shorthands[normalizer]
        return f"{t}{sh}" if sh!="" else ""

    def formatted_endunits(self, column, normalizer):
        if column not in self._udeets[normalizer]:
            #logger.debug(f"Normalizer: {normalizer} does not interact with {column}")
            return ""
        eu=self._udeets[normalizer][column]['end_units']\
            .replace("*",r'$\cdot$').replace("ohm",r"$\Omega$")\
            .replace("u",r"$\mu$").replace("$$","")
        return eu

    def normalizer_columns(self):
        return [k for k in self._udeets if k != 'None']

    @property
    def norm_options(self):
        return list(self._udeets.keys())



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
                df[[x]+yheaders_in_subtab+bystanders] \
                    .rename(columns={yheader:yheader[1:].split("@")[0] \
                                     for yheader in yheaders_in_subtab}) \
                    .assign(**{swv:v,'SweepDir':d}))
    if len(subtabs):
        return pd.concat(subtabs)
    else:
        return pd.DataFrame({k:[] for k in [x]+ys+bystanders})


def VTCC(I, V, icc, itol=1e-14):
    logI=np.log(np.abs(I)+itol)
    logicc=np.log(icc)

    ind_aboves=np.argmax(logI>logicc, axis=1)
    ind_belows=logI.shape[1]-np.argmax(logI[:,::-1]<logicc, axis=1)-1
    valid_crossing=(ind_aboves==(ind_belows+1))

    allinds=np.arange(len(I))
    lI1,lI2=logI[allinds,ind_belows],logI[allinds,ind_aboves]
    V1,V2=V[allinds,ind_belows],V[allinds,ind_aboves]

    slope=(lI2-lI1)/(V2-V1)
    Vavg=(V1+V2)/2
    lIavg=(lI1+lI2)/2

    VTcc=Vavg+(logicc-lIavg)/slope
    VTcc[~valid_crossing]=np.NaN

    return VTcc
