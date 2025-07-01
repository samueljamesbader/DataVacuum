from __future__ import annotations
import functools
from typing import TYPE_CHECKING, Any, Optional
from datavac.database.db_structure import DBSTRUCT, sql_to_pd_types
from datavac.util.logging import time_it
from datavac.util.util import returner_context
from sqlalchemy import Connection, select, Select, Column, Table
from datavac.measurements.measurement_group import MeasurementGroup
from datavac.io.measurement_table import MultiUniformMeasurementTable

if TYPE_CHECKING:
    import pandas as pd

def joined_select_from_dependencies(columns:Optional[list[str]],absolute_needs:list[Table],
                 table_depends:dict[Table,list[Table]], pre_filters:dict[str,list],
                 join_hints:dict[Table,str]={}) -> tuple[Select, dict[str, str]]:
    """ Creates an SQL Alchemy Select joining the tables needed to get the desired factors

    Args:
        columns: list of column names to be selected (or None to select all columns from table_depends)
        absolute_needs: list of tables which must be included in the join
        table_depends: mapping of dependencies which tables have on other tables to be joined
        pre_filters: filters (column name to list of allowed values) to apply to the data 
        join_hints: mapping of table to join clause to use when joining that table
            (generally not needed; if tables have foreign keys, sqlalchemy will infer)

    Returns:
        An SQLAlchemy Select and a dictionary mapping column names to their intended pandas types
    """
    all_cols=[c for tab in table_depends for c in tab.columns]
    def get_col(cname) -> Column:
        try: return next(c for c in all_cols if c.name==cname)
        except StopIteration:
            raise Exception(f"Couldn't find column {cname} among {[c.name for c in all_cols]}")
    def apply_wheres(s):
        for pf,values in pre_filters.items():
            s=s.where(get_col(pf).in_(values))
        return s
    if columns is not None:
        cols:list[Column]=[get_col(f) for f in columns]
    else:
        col_names = set()
        cols: list[Column] = []
        for tab in table_depends:
            for c in tab.columns:
                if c.name not in col_names:
                    cols.append(c); col_names.add(c.name)
    def apply_joins():
        ordered_needed_tables=[]
        need_queue=set(absolute_needs+[f.table for f in cols]+[get_col(pf).table for pf in pre_filters])
        while len(need_queue):
            for n in need_queue.copy():
                further_needs=table_depends[n]
                if n in ordered_needed_tables:
                    need_queue.remove(n)
                elif all(pn in ordered_needed_tables for pn in further_needs):
                    ordered_needed_tables.append(n)
                    need_queue.remove(n)
                else: need_queue|=set(further_needs)
        return functools.reduce((lambda x,y: x.join(y,onclause=join_hints.get(y,None))),ordered_needed_tables)
    sel=apply_wheres(select(*cols).select_from(apply_joins()))
    dtypes={c.name:sql_to_pd_types[c.type.__class__] for c in cols
                                             if c.type.__class__ in sql_to_pd_types}
    return sel, dtypes

def get_table_depends_and_hints_for_meas_group(meas_group: MeasurementGroup, include_sweeps: bool = True)\
            -> tuple[dict[Table, list[Table]], dict[Table, str]]:
        """Returns the table dependencies and join hints for a given measurement group.
        
        Args:
            meas_group: The measurement group to get the table dependencies for.
            include_sweeps: Whether to include the sweep table in the dependencies.
        Returns:
            A tuple containing:
            - A dictionary mapping tables to their dependencies (ie which tables they need to be joined with).
            - A dictionary mapping tables to their join hints (ie how to join them).
        """
        trove_name = meas_group.trove_name()

        table_depends={}
        table_depends[coretab:=(meastab:=DBSTRUCT().get_measurement_group_dbtables(meas_group.name)['meas'])]=[]
        table_depends[         (extrtab:=DBSTRUCT().get_measurement_group_dbtables(meas_group.name)['extr'])]=[meastab]
        table_depends[loadtab:=(DBSTRUCT().get_trove_dbtables(trove_name)['loads'])]=[coretab]
        table_depends[sampletab:=(DBSTRUCT().get_sample_dbtable())]=[loadtab]
        if include_sweeps and meas_group.involves_sweeps:
            table_depends[sweeptab:=(DBSTRUCT().get_measurement_group_dbtables(meas_group.name)['sweep'])]=[meastab]
        for ssr_name in meas_group.subsample_reference_names:
            # Note: just including the sample table here because in case the subsample reference has foreign keys
            # to some sample info, we need to make sure the sample info is merged before the subsample reference
            table_depends[DBSTRUCT().get_subsample_reference_dbtable(ssr_name)]=[coretab,DBSTRUCT().get_sample_dbtable()]

        # TODO: Uncomment when fnct and ot readded 
        #fnc_tables=[self._hat(fnct) for fnct in fnc_tables]
        #other_tables=[self._hat(ot) for ot in other_tables]
        #for fnct in fnc_tables: table_depends[fnct]=[self._mattab]
        #for ot in other_tables: table_depends[ot]=[self._mattab]

        return table_depends, {}


def _unstack_header_helper(data,unstacking_indices, drop_index=True) -> pd.DataFrame:
    """Unstacks the 'header' column in the data DataFrame, creating a wide format DataFrame with headers as columns.
    Args:
        data: The DataFrame containing the 'header' and 'sweep' columns to unstack.
        unstacking_indices: List of column names to use as indices for the unstacking.
        drop_index: Whether to drop the index after unstacking. Defaults to True.
    """
    import pandas as pd
    sweep_part=data[[*unstacking_indices,'header','sweep']] \
        .pivot(index=unstacking_indices,columns='header',values='sweep')
    other_part=data.drop(columns=['header','sweep']) \
        .drop_duplicates(subset=unstacking_indices).set_index(unstacking_indices)
    return pd.merge(sweep_part,other_part,how='inner',
                    left_index=True,right_index=True,validate='1:1').reset_index(drop=drop_index)

def _decode_sweeps(data: pd.DataFrame, meas_group: MeasurementGroup):
    """Decodes the sweeps in the data DataFrame for a given measurement group."""
    with time_it("Sweep decoding",threshold_time=.03):
        # This part is to handle the deprecated ONESTRING
        if (hasattr(meas_group,'ONESTRING') and meas_group.ONESTRING): # type: ignore
            from datavac.database.postgresql_binary_format import pd_to_pg_converters
            sweepconv = pd_to_pg_converters['STRING']
        # Else, it's a float32 sweep
        else:
            import numpy as np
            sweepconv = functools.partial(np.frombuffer, dtype=np.float32)
        data['sweep']=data['sweep'].map(sweepconv)

def get_data_for_reextr(meas_group: MeasurementGroup, samplename: Any,
                        on_no_data: str | None ='raise', conn: Optional[Connection] = None) -> MultiUniformMeasurementTable:
    """Retrieves data for re-extraction (ie measured data only, no extr data).
    
    Args:
        meas_group: The measurement group to retrieve data for.
        samplename: The name of the sample to retrieve data for.
        on_no_data: What to do if no data is found. Options are 'raise' (raise an exception) or None (return None).
        conn: Optional SQLAlchemy Connection to use for the query. If None, a read-only connection will be created.
    """
    import pandas as pd
    import numpy as np
    from datavac.config.data_definition import DDEF
    from datavac.database.db_connect import get_engine_ro
    from datavac.io.measurement_table import MultiUniformMeasurementTable, UniformMeasurementTable


    td, jh = get_table_depends_and_hints_for_meas_group(meas_group=meas_group, include_sweeps=True)
    td = {t: d for t, d in td.items() if 'Extr' not in t.name}
    sel, dtypes = joined_select_from_dependencies(columns=None, absolute_needs=list(td), table_depends=td,
                                                  pre_filters={DDEF().SAMPLE_COLNAME: [samplename]}, join_hints=jh)
    with (returner_context(conn) if conn is not None else get_engine_ro().connect()) as conn:
        data = pd.read_sql(con=conn, sql=sel, dtype=dtypes) # type: ignore

    if not(len(data)):
        match on_no_data:
            case 'raise':
                raise Exception(f"No data for re-extraction of {samplename} with measurement group {meas_group.name}")
            case None: return None # type: ignore

    umts=[]
    for rg, df in data.groupby("rawgroup"):
        if meas_group.involves_sweeps:
            _decode_sweeps(df, meas_group)
            df=_unstack_header_helper(df, ['loadid','measid'], drop_index=False)
            headers=[]
            for c in df.columns:
                if c in ['loadid','measid']: continue
                if c=='rawgroup': break
                headers.append(c)
        else:
            df=df.reset_index(drop=True)
            headers = []
        assert np.all(df.index == df['measid'])
        umts.append(UniformMeasurementTable(dataframe=df, headers=headers,
                                            meas_group=meas_group, meas_length=None))
    return MultiUniformMeasurementTable(umts)


def get_data_from_meas_group(meas_group: MeasurementGroup, scalar_columns:Optional[list[str]]=None,
                             include_sweeps:bool=False, unstack_headers:bool = False, conn:Optional[Connection]=None,
                             fnc_tables=[],other_tables=[],**factors) -> pd.DataFrame:
    import pandas as pd
    from datavac.database.db_connect import get_engine_ro

    if include_sweeps: assert meas_group.involves_sweeps, \
        f"Measurement group {meas_group.name} does not involve sweeps, but they were requested."
    td, jh = get_table_depends_and_hints_for_meas_group(meas_group=meas_group, include_sweeps=include_sweeps)
    sel, dtypes = joined_select_from_dependencies(columns=(scalar_columns+(['header','sweep'] if include_sweeps else [])
                                                           if scalar_columns is not None else None),
                                                  absolute_needs=list(td), table_depends=td,
                                                  pre_filters=dict(**factors,**({'header': include_sweeps} if isinstance(include_sweeps,list) else {})),
                                                  join_hints=jh)
    with (returner_context(conn) if conn is not None else get_engine_ro().connect()) as conn:
        df = pd.read_sql(con=conn, sql=sel, dtype=dtypes) # type: ignore

    if include_sweeps:
        _decode_sweeps(df, meas_group)
    if unstack_headers:
        df=_unstack_header_helper(df, ['loadid','measid'], drop_index=False)

    return df

def get_data(mg_or_an_name: str, scalar_columns: Optional[list[str]] = None,
             include_sweeps: bool = False, unstack_headers: bool = False,
             conn: Optional[Connection] = None, **factors) -> pd.DataFrame:
    """Retrieves data for a given measurement group or analysis name.
    
    Args:
        mg_or_an_name: The name of the measurement group or analysis to retrieve data for.
        scalar_columns: Optional list of scalar columns to include in the result.
        include_sweeps: Whether to include sweep data in the result.
        unstack_headers: Whether to unstack the 'header' column in the result.
        conn: Optional SQLAlchemy Connection to use for the query. If None, a read-only connection will be created.
        factors: Additional factors to filter the data by, eg factor1=['allowed_val1','allowed_val2']
    """
    from datavac.measurements.measurement_group import MeasurementGroup
    from datavac.config.data_definition import DDEF
    if (mg:=DDEF().measurement_groups.get(mg_or_an_name)) is not None:
        return get_data_from_meas_group(mg, scalar_columns=scalar_columns, include_sweeps=include_sweeps,
                                        unstack_headers=unstack_headers, conn=conn, **factors)
    else:
        raise NotImplementedError(f"Data retrieval for analysis {mg_or_an_name} is not implemented yet. "
                                  "Please implement it in get_data_from_analysis.")