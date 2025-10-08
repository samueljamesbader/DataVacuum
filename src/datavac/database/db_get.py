from __future__ import annotations

import functools
import graphlib
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from datavac.appserve.api import client_server_split
from datavac.config.data_definition import DDEF, HigherAnalysis
#from datavac.database.db_structure import DBSTRUCT, sql_to_pd_types
from datavac.util.dvlogging import time_it
from datavac.util.util import returner_context
#from sqlalchemy import INTEGER, Connection, select, Select, Column, Table, values
from datavac.measurements.measurement_group import MeasurementGroup

if TYPE_CHECKING:
    import pandas as pd
    from pandas import DataFrame
    from sqlalchemy import Connection, Table, Column, Select
    from datavac.io.measurement_table import MultiUniformMeasurementTable
else:
    # Otherwise Pydantic's validate_call won't be able to parse the types
    Connection=Any
    DataFrame=Any

def joined_select_from_dependencies(columns:Optional[list[str]],absolute_needs:list[Table],
                 table_depends:dict[Table,list[Table]], pre_filters:Mapping[str,Sequence],
                 join_hints:dict[Table,str]={},order_by:Optional[list[str]]=None) -> tuple[Select, dict[str, str]]:
    """ Creates an SQL Alchemy Select joining the tables needed to get the desired factors

    Args:
        columns: list of column names to be selected (or None to select all columns from table_depends)
        absolute_needs: list of tables which must be included in the join
        table_depends: mapping of dependencies which tables have on other tables to be joined
        pre_filters: filters (column name to list of allowed values) to apply to the data 
        join_hints: mapping of table to join clause to use when joining that table
            (generally not needed; if tables have foreign keys, sqlalchemy will infer)
        order_by: list of column names to order the results by (or None for no ordering)

    Returns:
        An SQLAlchemy Select and a dictionary mapping column names to their intended pandas types
    """
    from sqlalchemy import select
    from datavac.database.db_structure import DBSTRUCT, sql_to_pd_types
    ts = graphlib.TopologicalSorter(table_depends)
    ordered_tables = list(ts.static_order())

    all_cols=[c for tab in ordered_tables for c in tab.columns]
    def get_col(cname) -> Column:
        try: return next(c for c in all_cols if c.name==cname)
        except StopIteration:
            raise Exception(f"Couldn't find column {cname} among {[c.name for c in all_cols]}")
    def apply_wheres(s):
        for pf,values in pre_filters.items():
            s=s.where(get_col(pf).in_(values))
        return s
    if columns is not None:
        cols:list[Column]=[get_col(f) for f in [*columns,*(order_by if order_by is not None else [])]]
    else:
        col_names = set()
        cols: list[Column] = []
        for tab in ordered_tables:
            for c in tab.columns:
                if c.name not in col_names:
                    cols.append(c); col_names.add(c.name)

    starter_needs =  [t for t in ordered_tables if t in absolute_needs
                              or t in [f.table for f in cols]
                              or t in [get_col(pf).table for pf in pre_filters]]
    
    from datavac.util.dag import include_all_descendants
    all_needs = include_all_descendants(starter_needs, table_depends)
    ordered_needed_tables = [t for t in ordered_tables if t in all_needs]
    thejoin=functools.reduce((lambda x,y: x.join(y,onclause=join_hints.get(y,None))),ordered_needed_tables) # type: ignore
    sel=apply_wheres(select(*[c for c in cols if columns is None or c.name in columns]).select_from(thejoin))
    if order_by is not None:
        sel=sel.order_by(*[get_col(c).asc() for c in order_by])
    dtypes={c.name:sql_to_pd_types[c.type.__class__] for c in cols
                                             if c.type.__class__ in sql_to_pd_types
                                             and ((columns is None) or (c.name in columns))}
    return sel, dtypes

def get_table_depends_and_hints_for_meas_group(meas_group: MeasurementGroup, include_sweeps: bool = True, sample_descriptors: list[str]= [] )\
            -> tuple[dict[Table, list[Table]], dict[Table, str]]:
        """Returns the table dependencies and join hints for a given measurement group.
        
        Args:
            meas_group: The measurement group to get the table dependencies for.
            include_sweeps: Whether to include the sweep table in the dependencies.
        Returns:
            A tuple containing:
            - A dictionary mapping tables to their dependencies (ie which jtables they need to be joined with).
            - A dictionary mapping tables to their join hints (ie how to join them).
        """
        from datavac.database.db_structure import DBSTRUCT, sql_to_pd_types
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

        for sd_name in sample_descriptors: table_depends[DBSTRUCT().get_sample_descriptor_dbtable(sd_name)]=[sampletab]

        return table_depends, {}
def get_table_depends_and_hints_for_analysis(an: HigherAnalysis, sample_descriptors: list[str] = [])\
            -> tuple[dict[Table, list[Table]], dict[Table, str]]:
        """Returns the table dependencies and join hints for a given higher analysis.
        
        Args:
            an: The analysis to get the table dependencies for.
        Returns:
            A tuple containing:
            - A dictionary mapping tables to their dependencies (ie which tables they need to be joined with).
            - A dictionary mapping tables to their join hints (ie how to join them).
        """

        from datavac.database.db_structure import DBSTRUCT, sql_to_pd_types
        table_depends={}
        table_depends[coretab:=(anlstab:=DBSTRUCT().get_higher_analysis_dbtables(an.name)['anls'])]=[]
        table_depends[         (aidttab:=DBSTRUCT().get_higher_analysis_dbtables(an.name)['aidt'])]=[coretab]
        table_depends[sampletab:=(DBSTRUCT().get_sample_dbtable())]=[coretab]
        for ssr_name in an.subsample_reference_names:
            # Note: just including the sample table here because in case the subsample reference has foreign keys
            # to some sample info, we need to make sure the sample info is merged before the subsample reference
            table_depends[DBSTRUCT().get_subsample_reference_dbtable(ssr_name)]=[coretab,DBSTRUCT().get_sample_dbtable()]

        for sd_name in sample_descriptors: table_depends[DBSTRUCT().get_sample_descriptor_dbtable(sd_name)]=[sampletab]

        return table_depends, {}

def get_available_columns_for_mgoa(mgoa_name: str) -> list[str]:
    """Returns a list of available columns for a given measurement group or higher analysis name."""
    from datavac.config.data_definition import DDEF
    mg = DDEF().measurement_groups.get(mgoa_name)
    if mg is not None:
        td, _ = get_table_depends_and_hints_for_meas_group(mg, include_sweeps=False)
        return [c.name for t in td for c in t.columns]
    an = DDEF().higher_analyses.get(mgoa_name)
    if an is not None:
        td, _ = get_table_depends_and_hints_for_analysis(an)
        return [c.name for t in td for c in t.columns]
    raise ValueError(f"Measurement group or analysis '{mgoa_name}' not found in data definition.")

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

def get_data_as_mumt(meas_group: MeasurementGroup, samplename: Any, include_sweeps: bool = True, include_extr: bool = True,
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


    td, jh = get_table_depends_and_hints_for_meas_group(meas_group=meas_group, include_sweeps=(include_sweeps and meas_group.involves_sweeps))
    pre_filters={DDEF().SAMPLE_COLNAME: [samplename]}
    if not include_extr:
        td = {t: d for t, d in td.items() if 'Extr' not in t.name}
        if include_sweeps and meas_group.involves_sweeps:
            pre_filters['israw'] = [True]
    sel, dtypes = joined_select_from_dependencies(columns=None, absolute_needs=list(td), table_depends=td,
                                                  pre_filters=pre_filters, join_hints=jh,
                                                  order_by=['measid'])
    with (returner_context(conn) if conn is not None else get_engine_ro().connect()) as conn:
        data = pd.read_sql(con=conn, sql=sel, dtype=dtypes) # type: ignore
        assert data['loadid'].nunique() <= 1

    if not(len(data)):
        match on_no_data:
            case 'raise':
                raise Exception(f"No data available for {samplename} with measurement group {meas_group.name}")
            case None: return None # type: ignore

    umts=[]
    checking_measids=True # Can remove this check later for performance
    if checking_measids: all_measids=[]
    for rg, df in data.groupby("rawgroup"):
        if include_sweeps and meas_group.involves_sweeps:
            _decode_sweeps(df, meas_group)
            scalar_columns=list(df.columns)
            df=_unstack_header_helper(df, ['loadid','measid'], drop_index=False)
            headers=[c for c in df.columns if c not in scalar_columns]
        else:
            df=df.reset_index(drop=True)
            headers = []
        if checking_measids: all_measids.extend(df['measid'].to_list())
        df=df.drop(columns=['measid','rawgroup'], errors='ignore')
        umts.append(UniformMeasurementTable(dataframe=df, headers=headers,
                                            meas_group=meas_group, meas_length=None)) # type: ignore
    mumt=MultiUniformMeasurementTable(umts)
    if checking_measids: 
        df=mumt._dataframe
        assert np.all(df.index == all_measids),\
            f"Indices for {df.index} does not match measids {df['measid'].iloc[0]}..{df['measid'].iloc[-1]} while getting data for {meas_group.name}, {samplename}"
    return mumt


def get_data_from_meas_group(
    meas_group: MeasurementGroup,
    scalar_columns: Optional[list[str]] = None,
    include_sweeps: bool = False,
    unstack_headers: bool = False,
    conn: Optional[Connection] = None,
    sample_descriptors: list[str]=[],
    ensure_consistent_order: bool = False,
    **factors
) -> 'pd.DataFrame':
    import pandas as pd
    from datavac.database.db_connect import get_engine_ro

    if include_sweeps: assert meas_group.involves_sweeps, \
        f"Measurement group {meas_group.name} does not involve sweeps, but they were requested."
    assert not ((not include_sweeps) and unstack_headers), \
        f"Unstacking headers only makes sense when sweeps are included in the data."

    td, jh = get_table_depends_and_hints_for_meas_group(meas_group=meas_group, include_sweeps=include_sweeps, sample_descriptors=sample_descriptors)
    sel, dtypes = joined_select_from_dependencies(columns=(scalar_columns\
                                                                +(['header','sweep']  if include_sweeps  else [])\
                                                                +(['loadid','measid'] if unstack_headers else [])
                                                            if scalar_columns is not None else None),
                                                  absolute_needs=list(td), table_depends=td,
                                                  pre_filters=dict(**factors,**({'header': include_sweeps} if isinstance(include_sweeps,list) else {})),
                                                  join_hints=jh, order_by=['loadid','measid'] if ensure_consistent_order else None)
    with (returner_context(conn) if conn is not None else get_engine_ro().connect()) as conn:
        #df = pd.read_sql(con=conn, sql=sel, dtype=dtypes) # type: ignore
        #df = pd.read_sql(con=conn, sql, dtype=dtypes) # type: ignore # sometimes results in string without encoding TypeError when querying with split-tables
        df = pd.read_sql(con=conn, sql=str(sel.compile(conn,compile_kwargs={'literal_binds':True})), dtype=dtypes) # type: ignore

    if include_sweeps:
        _decode_sweeps(df, meas_group)
    if unstack_headers:
        df=_unstack_header_helper(df, ['loadid','measid'], drop_index=False)

    return df

@client_server_split("get_sweeps_for_jmp", return_type="pd", split_on="direct_db_access")
def get_sweeps_for_jmp(mg_name: str, loadids: list[int], measids: list[int],
                       only_sweeps: Optional[list[str]]=None) -> DataFrame:
    import pandas as pd; import numpy as np
    from datavac.config.data_definition import DDEF
    from datavac.database.db_connect import get_engine_ro
    from sqlalchemy import select, values, INTEGER, Column
    mg= DDEF().measurement_groups[mg_name]
    sweeptab=mg.dbtable('sweep')
    ids_cte=select(values(Column('loadid', INTEGER),Column('measid', INTEGER),name="lm_id_tab")\
                   .data(list(zip(loadids,measids)))).cte()
    
    sel=select(sweeptab.c.loadid, sweeptab.c.measid, sweeptab.c.sweep, sweeptab.c.header) \
        .select_from(sweeptab.join(ids_cte, (sweeptab.c.loadid==ids_cte.c.loadid) & (sweeptab.c.measid==ids_cte.c.measid)))
    if only_sweeps is not None:
        sel=sel.where(sweeptab.c.header.in_(only_sweeps))

    with get_engine_ro().connect() as conn:
        df = pd.read_sql(con=conn, sql=sel, dtype={'sweep': 'object', 'header': 'string'})
        
    _decode_sweeps(df, mg)
    df=_unstack_header_helper(df, ['loadid','measid'], drop_index=False)
    if not len(df): return df
    data = df; del df

    # Infer names of X and Y columns and sweep variables
    from datavac.util.tables import stack_multi_sweeps
    from datavac.util.util import only
    headers=[k for k in data.columns if k not in ['loadid','measid']]
    if all(('@' not in k) for k in headers):
        # Assume column names of form X, Y... doesn't actually matter which is independent, so say first
        # actually, it does matter in one situation: if X is something that might be NaN instead of array,
        # then the data.where() below will have an issue, so let's try to pick a good column for now.
        x=next((h for h in headers if h.lower() in ['x','time']),headers[0])
        swvs=[]
        ys_withdir=[h for h in headers if h!=x]
    else:
        # Assume column names of form X, fY1@SWVR=val, fY2@SWVR=val ...
        possible_xs=[k for k in headers if '@' not in k]
        x=only(possible_xs,f"None or multiple possible x values in {possible_xs}")
        swvs=list(set([eq.split("=")[0] for k in headers if '@' in k for eq in k.split("@")[1].split(",")]))
        ys_withdir=[k.split("@")[0] for k in headers if '@' in k]
    directed=all(y[0] in ('f','r') for y in ys_withdir)
    ys=list(set([y[1:] for y in ys_withdir] if directed else ys_withdir))
    ys=[y for y in ys if y not in swvs]

    # Restack the data so different sweep variables are columns rather than parts of headers
    data=stack_multi_sweeps(data,x=x,ys=ys,swvs=swvs,restrict_dirs=(('f','r') if directed else ('',))).reset_index(drop=True)

    # The above (both the unstack and the stack each) results in NaN values where there is no data for a given sweep
    # eg for IdVg if sometimes the source current is measured and sometimes not,
    # then there will be NaNs in the I_S column.
    # In order to be able to explode data, we will want to replace each NaN with an *array* of NaN's
    # of the same length as the independent variable.
    data=data.where(data.notna(),pd.Series([np.full(len(c),np.nan) for c in data[x]]),axis=0)
    try:
        data = data.explode([x,*ys],ignore_index=True)\
            [['loadid','measid',*[k for k in data.columns if k not in ['loadid','measid']]]]
    except:
        print({y:len(data.iloc[0][y]) for y in ys})
        raise
    return data

def get_data_from_analysis(an: HigherAnalysis, scalar_columns:Optional[list[str]]=None,
                           conn:Optional[Connection]=None, ensure_consistent_order: bool=False,
                           sample_descriptors:list[str]=[],**factors) -> pd.DataFrame:
    import pandas as pd
    from datavac.database.db_connect import get_engine_ro
    td, jh = get_table_depends_and_hints_for_analysis(an, sample_descriptors=sample_descriptors)
    sel, dtypes = joined_select_from_dependencies(columns=scalar_columns,
                                                  absolute_needs=list(td), table_depends=td,
                                                  pre_filters=factors, join_hints=jh,
                                                  order_by=['anlsid','anlssubid'] if ensure_consistent_order else None)
    with (returner_context(conn) if conn is not None else get_engine_ro().connect()) as conn:
        try:
            #df = pd.read_sql(con=conn, sql, dtype=dtypes) # type: ignore # sometimes results in string without encoding TypeError when querying with split-tables
            df = pd.read_sql(con=conn, sql=str(sel.compile(conn,compile_kwargs={'literal_binds':True})), dtype=dtypes) # type: ignore
        except TypeError as e:
            import pdb; pdb.set_trace()
            raise e
    return df
    
@client_server_split(method_name="get_data", return_type='pd')
def get_data(mg_or_an_name: str, scalar_columns: Optional[list[str]] = None,
             include_sweeps: bool|list[str] = False, unstack_headers: bool = False,
             conn: Optional[Connection] = None, ensure_consistent_order:bool=False,
             sample_descriptors:list[str]=[],**factors) -> DataFrame:
    """Retrieves data for a given measurement group or analysis name.
    
    Args:
        mg_or_an_name: The name of the measurement group or analysis to retrieve data for.
        scalar_columns: Optional list of scalar columns to include in the result.
        include_sweeps: Whether to include sweep data in the result.
        unstack_headers: Whether to unstack the 'header' column in the result.
        conn: Optional SQLAlchemy Connection to use for the query. If None, a read-only connection will be created.
        ensure_consistent_order: Whether to ensure a consistent order of results by ordering by ID columns.
        sample_descriptors: List of sample descriptor tables to include in the result.
        factors: Additional factors to filter the data by, eg factor1=['allowed_val1','allowed_val2']
    """
    from datavac.measurements.measurement_group import MeasurementGroup
    from datavac.config.data_definition import DDEF
    if (mg:=DDEF().measurement_groups.get(mg_or_an_name)) is not None:
        return get_data_from_meas_group(mg, scalar_columns=scalar_columns, include_sweeps=include_sweeps,
                                        unstack_headers=unstack_headers, conn=conn,
                                        ensure_consistent_order=ensure_consistent_order,
                                        sample_descriptors=sample_descriptors, **factors)
    elif (an:=DDEF().higher_analyses.get(mg_or_an_name)) is not None:
        assert not include_sweeps, \
            "Higher analyses do not support sweeps, please set include_sweeps=False"
        assert not unstack_headers, \
            "Higher analyses do not support unstacking headers, please set unstack_headers=False"
        return get_data_from_analysis(an, scalar_columns=scalar_columns, conn=conn,
                                      ensure_consistent_order=ensure_consistent_order,
                                      sample_descriptors=sample_descriptors,**factors)
    else:
        raise ValueError(f"Measurement group or analysis '{mg_or_an_name}' not found in data definition.")

@client_server_split(method_name="get_factors", return_type='ast')
def get_factors(meas_group_or_analysis: str,factor_names:list[str],pre_filters:Mapping[str,Sequence]={},
                sample_descriptors:list[str]=[]):
    import pandas as pd
    mgoa_name=meas_group_or_analysis; del meas_group_or_analysis

    # Get the relevant table dependencies and join hints
    if mgoa_name in DDEF().measurement_groups:
        mg=DDEF().measurement_groups[mgoa_name]
        coretab=mg.dbtable('meas')
        td, jh = get_table_depends_and_hints_for_meas_group(mg, include_sweeps=False, sample_descriptors=sample_descriptors)
    elif mgoa_name in DDEF().higher_analyses:
        an=DDEF().higher_analyses[mgoa_name]
        coretab=an.dbtables('aidt')
        td, jh = get_table_depends_and_hints_for_analysis(an, sample_descriptors=sample_descriptors)
    else:
        raise ValueError(f"Measurement group or analysis '{mgoa_name}' not found in data definition.")
    
    # Define a select with the pre_filtered data and set it up as a CTE named tmp
    sel, dtypes = joined_select_from_dependencies(columns=factor_names,
                                                  absolute_needs=[coretab], table_depends=td,
                                                  pre_filters=pre_filters, join_hints=jh)
    query="WITH tmp AS (" + str(sel.compile(compile_kwargs={"literal_binds": True})) + ")\n"

    # Now we're going to build a series of select statements that query the distinct values of each factor
    # (but we use GROUP BY instead of DISTINCT ON because it's wayyy faster)
    # and each of those selects also defines a row number within its selection (ie 0,1,2,3 for each unique value)
    # and that row number will be called "ind___{factor_name}"
    # Then we do a full outer join on those indices, so the resulting table will have two columns for each factor,
    # one column that's just indices and one column that's the unique factor values.  Because it's a full outer
    # join, it will be as long as the factor with the most unique values.  But the other factors will have NULLs,
    # and will also have NULL indices, so in post, we can easily grab all the unique values for each factor,
    # ignoring rows where the index for that factor is NULL
    query+="SELECT * FROM \n"
    for i,f in enumerate(factor_names):
        # Note: by the time we get here, self.joined_table has already validated that f is the name
        # of a column in the joined table, so these names are guaranteed to be valid
        f=f.replace('%','%%') # escape % in case it's in the name
        query+=f"""(SELECT ROW_NUMBER() over (order by null) as "ind___{f}", tmp."{f}" """\
                                                        f""" FROM tmp GROUP BY tmp."{f}") t{i}\n"""
        if i!=0: query+=f""" ON t0."ind___{factor_names[0]}"=t{i}."ind___{f}" """
        if i!=len(factor_names)-1: query+=" FULL OUTER JOIN "

    # Execute it
    from datavac.database.db_connect import get_engine_ro
    with get_engine_ro().begin() as conn:
        with time_it("Executing SQL in get_factors",threshold_time=.01):
            records=pd.read_sql(query,conn,dtype=dtypes)
    if not len(records): return {f:set() for f in factor_names}
    else: return {f:set(records.loc[records[f'ind___{f}'].notna(),f].to_list()) for f in factor_names}

@client_server_split(method_name="get_mgoa_names", return_type='ast')
def get_mgoa_names():
    return sorted(list(DDEF().measurement_groups.keys()) + list(DDEF().higher_analyses.keys()))

@client_server_split(method_name="get_available_columns", return_type='ast', split_on="direct_db_access")
def get_available_columns(mgoa_name: str) -> list[str]:

    # Get the relevant table dependencies and join hints
    if mgoa_name in DDEF().measurement_groups:
        mg=DDEF().measurement_groups[mgoa_name]
        coretab=mg.dbtable('meas')
        td, jh = get_table_depends_and_hints_for_meas_group(mg, include_sweeps=False)
    elif mgoa_name in DDEF().higher_analyses:
        an=DDEF().higher_analyses[mgoa_name]
        coretab=an.dbtables('aidt')
        td, jh = get_table_depends_and_hints_for_analysis(an)
    else:
        raise ValueError(f"Measurement group or analysis '{mgoa_name}' not found in data definition.")
    
    # Define a select with the pre_filtered data and set it up as a CTE named tmp
    sel, dtypes = joined_select_from_dependencies(None,
                    absolute_needs=[coretab], table_depends=td, pre_filters={}, join_hints=jh)
    return list(sorted(sel.selected_columns.keys()))