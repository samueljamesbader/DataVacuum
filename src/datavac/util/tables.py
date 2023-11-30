import pandas as pd
from pandas import DataFrame


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


def check_dtypes(dataframe:DataFrame):
    assert 'in_place' not in dataframe.columns, "There's a column named 'in_place', this is probably a mistake..."
    for c,dtype in dataframe.dtypes.items():
        assert str(dtype)!='object', \
            f"Column '{c}' has dtype object! Usually this is supposed to be string or some nullable type"
    return dataframe
