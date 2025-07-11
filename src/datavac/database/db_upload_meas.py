from __future__ import annotations

from datetime import datetime
from functools import reduce
from itertools import groupby
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, cast
from datavac.database.db_connect import get_engine_rw
from datavac.database.postgresql_upload_utils import upload_binary, upload_csv
from datavac.measurements.measurement_group import MeasurementGroup
from datavac.trove import Trove
from datavac.util.util import asnamedict, only, returner_context
from sqlalchemy import VARCHAR, Column, Connection, delete, literal, select, text, values
from sqlalchemy.dialects.postgresql import insert as pgsql_insert, BYTEA, TIMESTAMP

from datavac.util.dvlogging import logger
from datavac.config.project_config import PCONF
from datavac.database.db_structure import DBSTRUCT
from datavac.config.data_definition import DDEF, HigherAnalysis

if TYPE_CHECKING:
    from datavac.io.measurement_table import MeasurementTable, MultiUniformMeasurementTable
    import pandas as pd

def enter_sample(conn: Connection, **sample_info: dict[str,Any]):
    """ Enter a sample into the Materials table, or update it if it already exists.

    Note:
        Does not commit the transaction.

    Args:
        conn: The database connection to use.
        sample_info: A dictionary about the sample, keyed by DataDefinition.sample_info_columns
    """
    samplename_col=PCONF().data_definition.sample_identifier_column.name
    sampletab=DBSTRUCT().get_sample_dbtable()
    update_info=sample_info.copy()
    #update_info.update(date_user_changed=datetime.now())
    sampleid=conn.execute(pgsql_insert(sampletab)\
                       .values(**update_info)\
                       .on_conflict_do_update(index_elements=[samplename_col],set_=update_info)\
                       .returning(sampletab.c.sampleid))\
                .all()[0][0]
    return sampleid

def delete_sample(conn: Connection, sample_info: dict[str,Any]):
    """ Delete a sample from the Materials table.

    Note:
        Does not commit the transaction.
        
    Args:
        conn: The database connection to use.
        sample_info: A dictionary about the sample, keyed by DataDefinition.sample_info_columns
            Note that only the sample identifier column is used to identify the sample. The rest 
            of the sample_info is ignored.
    """
    samplename_col=PCONF().data_definition.sample_identifier_column.name
    sampletab=DBSTRUCT().get_sample_dbtable()
    res=conn.execute(delete(sampletab)\
                     .where(sampletab.c[samplename_col]==sample_info[samplename_col]))
                     #.returning(sampletab.c.date_user_changed)).all()
    #if len(res): return res[0][0]

def delete_prior_loads(trove: Trove, sample_info: Optional[dict[str,Any]],
                       conn: Connection, only_meas_groups:Optional[list[str]]=None):
    """ Delete prior loads for a sample in the Loads table and place them into the Reload table.

    Args:
        trove: The Trove for which the loads should be deleted.
        sample_info: A dictionary about the sample, keyed by DataDefinition.sample_info_columns
            Note that only the sample identifier column is used to identify the sample. The rest 
            of the sample_info is ignored.  Supply None to delete all loads for all samples.
        conn: The database connection to use.
        only_meas_groups: If provided, only loads for these measurement groups will be deleted.
            If None, all loads for the sample within this trove will be deleted.
    """
    samplename_col=PCONF().data_definition.SAMPLE_COLNAME
    sampletab=DBSTRUCT().get_sample_dbtable()
    loadtab=DBSTRUCT().get_trove_dbtables(trove.name)['loads']
    relotab=DBSTRUCT().get_trove_dbtables(trove.name)['reload']
    copy_cols=[c for c in loadtab.c if c.name!='loadid']
    copy_colnames=[c.name for c in copy_cols]

    # Create a CTE by deleting the loads for the sample
    statement=delete(loadtab).returning(*copy_cols)

    # If sample_info is not None, restrict the deletion to that sample
    if sample_info is not None:
         statement=statement \
             .where(sampletab.c[samplename_col]==sample_info[samplename_col]) \
             .where(sampletab.c.sampleid==loadtab.c.sampleid)\
    
    # If only_meas_groups is provided, restrict the deletion to those measurement groups
    if only_meas_groups is not None:
        statement=statement.where(loadtab.c.MeasGroup.in_(only_meas_groups))

    # Copy those deleted rows into the Reload table
    anon=statement.cte()
    statement=pgsql_insert(relotab).from_select(copy_colnames,
                select(*[anon.c[c] for c in copy_colnames]).select_from(anon))
    
    # Execute it all
    conn.execute(statement)

def upload_measurement(trove: Trove, sampleload_info: Mapping[str,Any],
                       data_by_meas_group: Mapping[str, MeasurementTable],
                       only_meas_groups: Optional[list[str]] = None) -> dict[str,int]:
    sampletab=DBSTRUCT().get_sample_dbtable()
    loadtab=DBSTRUCT().get_trove_dbtables(trove.name)['loads']
    sample_info={k:v for k,v in sampleload_info.items()
                 if k in PCONF().data_definition.ALL_SAMPLE_COLNAMES}
    load_info={k:v for k,v in sampleload_info.items() if k in PCONF().data_definition.ALL_LOAD_COLNAMES(trove.name)}

    with get_engine_rw().begin() as conn:

        # Ensure sample exists
        sampleid = enter_sample(conn, **sample_info)

        # Delete prior loads for this sample (moving to ReLoad table)
        delete_prior_loads(trove, sample_info, conn, only_meas_groups=only_meas_groups)

        mg_to_loadid={}
        for mg_name, meas_data in data_by_meas_group.items():

            # Put an entry into the Loads table and get the loadid
            mg_to_loadid[mg_name]=loadid=conn.execute(pgsql_insert(loadtab)\
                                           .values(sampleid=sampleid,MeasGroup=mg_name,**load_info)\
                                           .returning(loadtab.c.loadid))\
                                    .all()[0][0]
            
            # Upload the measurement data
            meastab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['meas']
            content=meas_data.scalar_table.reset_index()\
                           .assign(loadid=loadid).rename(columns={'index':'measid'})
            for ssr_name in PCONF().data_definition.measurement_groups[mg_name].subsample_reference_names:
                PCONF().data_definition.subsample_references[ssr_name]\
                    .transform(content, sample_info=sample_info, conn=conn)
            upload_csv(content[list(meastab.c.keys())],
                       conn, DBSTRUCT().int_schema, meastab.name)
            
            # Upload the sweeps
            sstab=meas_data.get_stacked_sweeps()
            if len(sstab):

                import numpy as np
                def sweep_to_bytes(s: np.ndarray) -> bytes:
                    """Convert a sweep to bytes, handling both float32 and 'onestring' types."""
                    if s.dtype != np.float32:
                        raise TypeError(f"Sweep data type {s.dtype} for {mg_name} is not supported. Only float32 is supported.")
                    return s.tobytes()

                # This part is to handle the deprecated ONESTRING
                from datavac.database.postgresql_binary_format import pd_to_pg_converters
                sweepconv=(pd_to_pg_converters['STRING']) \
                    if (hasattr(mg_name,'ONESTRING') and mg_name.ONESTRING) else sweep_to_bytes # type: ignore
                # Convert to the appropriate format and upload
                upload_binary(
                    sstab.assign(loadid=loadid)[['loadid','measid','sweep','header']],
                    conn,DBSTRUCT().int_schema,DBSTRUCT().get_measurement_group_dbtables(mg_name)['sweep'].name,
                    override_converters={'sweep':sweepconv,'header':pd_to_pg_converters['STRING']}
                )
        
        relotab=DBSTRUCT().get_trove_dbtables(trove.name)['reload']
        reextab=DBSTRUCT().get_trove_dbtables(trove.name)['reextr']

        statements=[
            delete(relotab) \
                    .where(sampletab.c.sampleid==literal(sampleid))\
                    .returning(relotab.c.MeasGroup),
            *[pgsql_insert(reextab).from_select(['loadid'],select(literal(loadid)))
                for loadid in mg_to_loadid.values()]
        ]
        conn.execute(text(';'.join([str(s.compile(conn,compile_kwargs={'literal_binds':True})) for s in statements])))
        return mg_to_loadid

def delete_prior_extractions(trove: Trove, sampleload_info: Optional[Mapping[str,Any]], 
                             only_meas_groups: Optional[list[str]] = None,
                             conn: Optional[Connection] = None) -> dict[str,list[int]]:
    """ Delete prior extractions for a sample in the ReExtract table and place them into the ReExtract table.

    Args:
        trove: The Trove for which the extractions should be deleted.
        sampleload_info: A dictionary about the sample, keyed by DataDefinition.sample_info_columns
            Note that only the sample identifier column is used to identify the sample. The rest 
            of the sample_info is ignored.  Supply None to delete all extractions for all samples.
        only_meas_groups: If provided, only extractions for these measurement groups will be deleted.
            If None, all extractions for the sample within this trove will be deleted.
        conn: The database connection to use. If None, a new connection will be created.
    Returns:
        A dictionary mapping measurement group names to lists of loadids that were deleted.
    """
    # TODO: generalize for multi-trove system and remove the trove argument
    from datavac.database.db_structure import DBSTRUCT
    from sqlalchemy import delete, literal

    loadtab=DBSTRUCT().get_trove_dbtables(trove.name)['loads']
    with get_engine_rw().begin() as conn:

        # Select the relevant loadids by MeasGroup
        loadidsel = select(loadtab.c.MeasGroup,loadtab.c.loadid)

        # If sampleload_info is not None, restrict the selection to that sample
        if sampleload_info is not None:
            samplename_col=PCONF().data_definition.SAMPLE_COLNAME
            sampletab=DBSTRUCT().get_sample_dbtable()
            loadidsel=loadidsel.where(sampletab.c[samplename_col]==sampleload_info[samplename_col])\
                               .where(sampletab.c.sampleid==loadtab.c.sampleid)
            
        # If only_meas_groups is provided, restrict the selection to those measurement groups
        if only_meas_groups is not None:
            loadidsel=loadidsel.where(loadtab.c.MeasGroup.in_(only_meas_groups))
        
        # Roll up all the loadids
        statements= []
        mg_name_to_loadids = {k:[vi[1] for vi in v] for k,v in\
            groupby(sorted(conn.execute(loadidsel).all(),key=lambda x: x[0]),key=lambda x: x[0])}
        for mg_name, loadids in mg_name_to_loadids.items():
            # Delete the extractions for this loadid
            extrtab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['extr']
            statements.append(delete(extrtab).where(extrtab.c.loadid.in_(loadids)))

            # And put an entry into the ReExtract table
            reextab=DBSTRUCT().get_trove_dbtables(trove.name)['reextr']
            for loadid in loadids:
                statements.append(pgsql_insert(reextab).values(loadid=loadid).on_conflict_do_nothing())

        if len(statements):
            # Execute all the statements in a single transaction
            conn.execute(text(';'.join([str(s.compile(conn,compile_kwargs={'literal_binds':True})) for s in statements])))

        return mg_name_to_loadids                 
    

def upload_extraction(trove: Trove, samplename: Any,
                      data_by_meas_group: Mapping[str, MeasurementTable],
                      only_meas_groups: Optional[list[str]] = None,
                      skip_delete_use_these_loadids: dict[str,int] = {}):
    # TODO: generalize for multi-trove system and remove the trove argument

    mg_name_to_loadid=skip_delete_use_these_loadids.copy()
    need_to_delete_and_acquire_loadid=[mg_name for mg_name in (only_meas_groups or data_by_meas_group)
                                       if mg_name not in skip_delete_use_these_loadids]
    mg_name_to_loadid=dict(**mg_name_to_loadid,
                  **{k:only(v) for k,v in
                        delete_prior_extractions(trove, {DDEF().SAMPLE_COLNAME:samplename},
                            only_meas_groups=need_to_delete_and_acquire_loadid).items()})

    for mg_name in (only_meas_groups or data_by_meas_group):
        meas_data = data_by_meas_group[mg_name]
        
        # Upload the extraction data
        extrtab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['extr']
        content=meas_data.scalar_table.assign(loadid=mg_name_to_loadid[mg_name])
        if 'measid' not in content.columns: content=content.reset_index().rename(columns={'index':'measid'})
        with get_engine_rw().begin() as conn:
            upload_csv(content[list(extrtab.c.keys())],
                       conn, DBSTRUCT().int_schema, extrtab.name)
        
            reextab=DBSTRUCT().get_trove_dbtables(trove.name)['reextr']
            reantab=DBSTRUCT().get_higher_analysis_reload_table()
            sampletab=DBSTRUCT().get_sample_dbtable()
            needed_analyses = [(an.name,) for an in DDEF().get_meas_groups_dependent_graph()[DDEF().measurement_groups[mg_name]]
                                if isinstance(an, HigherAnalysis)]
            needed_analyses_cte=select(values(Column('analysis', VARCHAR),name="needed_analysis").data(needed_analyses)).cte()
            statements:list=[delete(reextab).where(reextab.c.loadid==mg_name_to_loadid[mg_name])]
            if len(needed_analyses):
                statements.append(
                    pgsql_insert(reantab).from_select(['sampleid','Analysis'],
                                                      select(sampletab.c.sampleid,needed_analyses_cte.c.analysis)\
                                                        .where(sampletab.c[DDEF().SAMPLE_COLNAME]==literal(samplename)))\
                                         .on_conflict_do_nothing())
            query_str = ';'.join([str(s.compile(conn,compile_kwargs={'literal_binds':True})) for s in statements])
            #print(query_str)
            conn.execute(text(query_str))
    return mg_name_to_loadid
        

def read_and_enter_data(trove_names: Optional[list[str]] = None,
                        only_meas_groups: Optional[list[str]] = None,
                        only_sampleload_info: dict[str,Sequence[Any]] = {},
                        info_already_known: dict[str,Any] = {}, 
                        kwargs_by_trove: dict[str,dict[str,Any]] = {}):
    trove_names = trove_names or list(PCONF().data_definition.troves.keys())
    for trove_name in trove_names:
        if trove_name not in PCONF().data_definition.troves:
            raise ValueError(f"Trove '{trove_name}' not found in data definition.")
        trove=PCONF().data_definition.troves[trove_name]
        sample_to_mg_to_data, sample_to_sampleloadinfo = trove.read(
            only_meas_groups=only_meas_groups, only_sampleload_info=only_sampleload_info,
            info_already_known=info_already_known, **(kwargs_by_trove.get(trove_name,{})))
        for sample, data_by_mg in sample_to_mg_to_data.items():
            assert PCONF().data_definition.SAMPLE_COLNAME in sample_to_sampleloadinfo[sample], \
                f"Sample info for '{sample}' does not have the sample identifier column "\
                f"'{PCONF().data_definition.SAMPLE_COLNAME}', just\n{sample_to_sampleloadinfo[sample]}" 
            mg_to_loadid=upload_measurement(trove, sample_to_sampleloadinfo[sample], data_by_mg,
                                            only_meas_groups=only_meas_groups)
            data_by_mg,mg_to_loadid=perform_and_enter_extraction(trove, samplename=sample,
                            only_meas_groups=list(data_by_mg),
                            pre_obtained_data_by_mg=data_by_mg,
                            pre_obtained_mg_to_loadid=mg_to_loadid)
            needed_analyses = list(set(an.name for mg in data_by_mg
                                      for an in DDEF().get_meas_groups_dependent_graph()[DDEF().measurement_groups[mg]]
                                        if isinstance(an, HigherAnalysis)))
            perform_and_enter_analysis(sample_info=sample_to_sampleloadinfo[sample], only_analyses=needed_analyses,
                                       pre_obtained_data_by_mgoa=data_by_mg, pre_obtained_mgoa_to_loadanlsid=mg_to_loadid,)

def perform_and_enter_extraction(trove: Trove, samplename: Any, only_meas_groups: list[str],
              pre_obtained_data_by_mg: dict[str,MultiUniformMeasurementTable]={},
              pre_obtained_mg_to_loadid: dict[str,int]={},
              conn: Optional[Connection] = None) -> tuple[dict[str,MultiUniformMeasurementTable],dict[str,int]]:
    """ Re-extract data for a sample.

    Caution: the tables in pre_obtained_data_by_mg will be modified in place by extraction.

    Args:
        samplename: The name of the sample to re-extract.
        only_meas_groups: Minimal list of measurement groups to re-extract.
            (Extractions which depend on these measurement groups will also be re-extracted).
        pre_obtained_data_by_mg: data already obtained for the measurement groups.  Any measurement groups
            in only_meas_groups but not supplied here will be obtained from the database.
        conn: The database connection to use. If None, a new connection will be created.
    """
    from datavac.database.db_get import get_data_as_mumt, get_data
    from datavac.database.db_upload_meas import upload_extraction
    from datavac.measurements.meas_util import perform_extraction

    # TODO: generalize for multi-trove system and remove the trove argument
    with (returner_context(conn) if conn is not None else get_engine_rw().begin()) as conn:

        data_by_mg = {}
        required_meas_groups = set(only_meas_groups)
        mg_retrieval_queue = [mg.name for mg in DDEF().get_meas_groups_topo_sorted() if mg.name in only_meas_groups]
        already_tried_no_data = set()
        should_reextract = set(only_meas_groups)

        # Go through the retrieval queue until emptied
        while len(mg_retrieval_queue):
            mg_name = mg_retrieval_queue.pop(0)

            # If this measurement group has already been located and has data, skip it
            if mg_name in data_by_mg: continue

            # If the measurement group has already been tried and has no data, skip unless it is required
            if (mg_name in already_tried_no_data):
                if mg_name in required_meas_groups:
                    raise ValueError(f"Measurement group '{mg_name}' is required but no data was obtained for it.")
                else: continue
            
            # Use the pre-obtained data if possible
            if mg_name in pre_obtained_data_by_mg:
                data_by_mg[mg_name] = data = pre_obtained_data_by_mg[mg_name]
            # Otherwise get the data for this measurement group
            else:
                # Raise error if a required measurement group has no data
                data = get_data_as_mumt(DDEF().measurement_groups[mg_name], conn=conn, samplename=samplename,
                                           on_no_data=('raise' if mg_name in required_meas_groups else None),
                                           include_extr=(mg_name not in should_reextract), include_sweeps=(mg_name in should_reextract))
                if (mg_name not in should_reextract) and (data is None) and (mg_name in required_meas_groups):
                    raise ValueError(f"Measurement group '{mg_name}' is required but no data was obtained for it.")
            if data is None: already_tried_no_data.add(mg_name)
            else:
                data_by_mg[mg_name] = data
                # If the group should be re-extracted, add its dependencies to the queue
                # and add its dependent groups to the queue and set of groups to re-extract
                if mg_name in should_reextract:
                    required_meas_groups|=set(DDEF().measurement_groups[mg_name].required_dependencies)
                    mg_retrieval_queue.extend(DDEF().measurement_groups[mg_name].required_dependencies)
                    mg_retrieval_queue.extend(DDEF().measurement_groups[mg_name].optional_dependencies)

                    for forward_mg in DDEF().get_meas_groups_dependent_graph()[DDEF().measurement_groups[mg_name]]:
                        if not isinstance(forward_mg, MeasurementGroup): continue
                        mg_retrieval_queue.append(forward_mg.name)
                        should_reextract.add(forward_mg.name)
        should_reextract = list(should_reextract) 
                
                
        # Perform the extraction
        perform_extraction({samplename: data_by_mg}, only_meas_groups=should_reextract)

        # Upload
        return data_by_mg,upload_extraction(trove, samplename=samplename,
                                 data_by_meas_group=data_by_mg, only_meas_groups=should_reextract,
                                 skip_delete_use_these_loadids=pre_obtained_mg_to_loadid)

def delete_prior_analyses(samplename: Any, only_analyses: Optional[list[str]] = None, conn: Optional[Connection] = None, can_skip_adding_to_rean: bool = False):
    sampletab = DBSTRUCT().get_sample_dbtable()
    statements=[]

    only_analyses = only_analyses or list(DDEF().higher_analyses.keys())
    assert only_analyses is not None
    if len(only_analyses) == 0: return

    if not can_skip_adding_to_rean:
        if samplename is None:
            isel:Select = reduce((lambda x,y: x.union_all(y)), # type: ignore
                          (select(an.dbtables('aidt').c.sampleid, literal(an_name))
                               for an_name, an in DDEF().higher_analyses.items() if an_name in only_analyses))
        else:
            isel:Select = reduce((lambda x,y: x.union_all(y)), # type: ignore
                          (select(an.dbtables('aidt').c.sampleid, literal(an_name))\
                            .where(an.dbtables('aidt').c.sampleid==sampletab.c.sampleid)\
                            .where(sampletab.c[PCONF().data_definition.SAMPLE_COLNAME]==literal(samplename))\
                               for an_name, an in DDEF().higher_analyses.items() if an_name in only_analyses))
        reantab=DBSTRUCT().get_higher_analysis_reload_table()
        statements.append(pgsql_insert(reantab).from_select(['sampleid','Analysis'],isel).on_conflict_do_nothing())
    
    for an_name in only_analyses:
        an = DDEF().higher_analyses[an_name]
        if samplename is None:
            statements.append(delete(an.dbtables('aidt')))
        else:
            statements.append(delete(an.dbtables('aidt'))\
                              .where(sampletab.c['sampleid']==an.dbtables('aidt').c['sampleid'])\
                              .where(sampletab.c[PCONF().data_definition.SAMPLE_COLNAME]==literal(samplename)))
    with (returner_context(conn) if conn is not None else get_engine_rw().begin()) as conn:
        q=';'.join([str(s.compile(conn,compile_kwargs={'literal_binds':True})) for s in statements])
        #print(q)
        conn.execute(text(q))


def upload_analysis(an: HigherAnalysis, sample_info: dict[str,Any], data: pd.DataFrame,
                    pre_obtained_mgoa_to_loadanlsid: dict[str,int|None]={},
                    conn: Optional[Connection] = None):
    samplename = sample_info[DDEF().SAMPLE_COLNAME]
    with (returner_context(conn) if conn is not None else get_engine_rw().begin()) as conn:
        delete_prior_analyses(samplename, only_analyses=[an.name], conn=conn, can_skip_adding_to_rean=True)
        idrefs={}
        for dep_name in [*an.required_dependencies, *an.optional_dependencies]:
            if dep_name in DDEF().measurement_groups:
                idrefs[f'loadid - {dep_name}'] = pre_obtained_mgoa_to_loadanlsid[dep_name]
            if dep_name in DDEF().higher_analyses:
                idrefs[f'anlsid - {dep_name}'] = pre_obtained_mgoa_to_loadanlsid[dep_name]
        sampletab=DBSTRUCT().get_sample_dbtable()
        SAMPLECOL=sampletab.c[PCONF().data_definition.SAMPLE_COLNAME]
        anlsid=conn.execute(pgsql_insert(an.dbtables('aidt'))\
                         .from_select(['sampleid', *idrefs.keys()],
                                select(sampletab.c['sampleid'], *[literal(v) for v in idrefs.values()])\
                                    .where(SAMPLECOL==literal(samplename)))\
                         .returning(an.dbtables('aidt').c.anlsid)).scalar_one()
        data=data.assign(anlsid=anlsid)
        for ssr_name in an.subsample_reference_names:
            PCONF().data_definition.subsample_references[ssr_name]\
                .transform(data, sample_info, conn=conn)
        upload_csv(data[[c.name for c in an.dbtables('anls').c]],conn, DBSTRUCT().int_schema, an.dbtables('anls').name)

        reantab=DBSTRUCT().get_higher_analysis_reload_table()
        statements:list=[
            delete(reantab).where(SAMPLECOL==literal(samplename))\
                .where(reantab.c['sampleid']==sampletab.c['sampleid'])\
                .where(reantab.c.Analysis==literal(an.name))]
        dpnt_analyses = [(an2.name,) for an2 in DDEF().get_analyses_dependent_graph()[an]]
        if len(dpnt_analyses):
            dpnt_analyses_cte=select(values(Column('analysis', VARCHAR),name="dpdt_analysis").data(dpnt_analyses)).cte()
            statements.append(pgsql_insert(reantab).from_select(['sampleid','Analysis'],
                                              select(sampletab.c.sampleid,dpnt_analyses_cte.c.analysis)
                                                .where(SAMPLECOL==literal(samplename)))\
                                     .on_conflict_do_nothing())
        query_str = ';'.join([str(s.compile(conn,compile_kwargs={'literal_binds':True})) for s in statements])
        #print(query_str)
        conn.execute(text(query_str))
        return anlsid


def perform_and_enter_analysis(sample_info:dict[str,Any], only_analyses: list[str],
              pre_obtained_data_by_mgoa: Mapping[str,MultiUniformMeasurementTable|pd.DataFrame]={},
              pre_obtained_mgoa_to_loadanlsid: dict[str,int]={},
              conn: Optional[Connection] = None):
    from datavac.database.db_get import get_data, get_data_as_mumt
    from datavac.io.measurement_table import MultiUniformMeasurementTable
    
    needs_analysis = set([DDEF().higher_analyses[an_name] for an_name in only_analyses])
    data_by_mgoa: dict[str, MeasurementTable|pd.DataFrame] = {}
    already_tried_no_data = set()
    mgoa_to_loadanlsid:dict[str,int|None]=pre_obtained_mgoa_to_loadanlsid.copy() # type: ignore
    samplename = sample_info[DDEF().SAMPLE_COLNAME]
    for an in DDEF().get_higher_analyses_topo_sorted():
        if an in needs_analysis:
            possible_to_perform = True
            for required,lst in [(True,an.required_dependencies) , (False, an.optional_dependencies)]:
                for dep_mgoa in lst:
                    if not possible_to_perform: break
                    if dep_mgoa not in data_by_mgoa:
                        if dep_mgoa in already_tried_no_data: data=None
                        else:
                            if dep_mgoa in pre_obtained_data_by_mgoa:
                                data = pre_obtained_data_by_mgoa[dep_mgoa]
                            else:
                                if dep_mgoa in DDEF().measurement_groups:
                                    data = get_data_as_mumt(DDEF().measurement_groups[dep_mgoa], conn=conn, samplename=samplename,
                                               on_no_data=('raise' if required else None), include_extr=True)
                                else:
                                    data = get_data(dep_mgoa, conn=conn, **{DDEF().SAMPLE_COLNAME:[samplename]}) # type: ignore
                                if data is None:
                                    already_tried_no_data.add(dep_mgoa)
                                    mgoa_to_loadanlsid[dep_mgoa]=None
                                else:
                                    mgoa_to_loadanlsid[dep_mgoa]=\
                                        int(only(cast(MultiUniformMeasurementTable,data).s['loadid'].unique()))\
                                            if dep_mgoa in DDEF().measurement_groups else\
                                        int(only(cast(pd.DataFrame,data)['anlsid'].unique()))
                        if data is not None: data_by_mgoa[dep_mgoa]=data
                        elif required:  possible_to_perform = False
            if possible_to_perform:
                logger.info(f"{an.name} analysis ({samplename})")
                data_by_mgoa[an.name]=an.analyze(
                    **{v:data_by_mgoa.get(k) for k,v in an.required_dependencies.items()},
                    **{v:data_by_mgoa.get(k) for k,v in an.optional_dependencies.items()},)
                for an2 in DDEF().get_analyses_dependent_graph()[an]: needs_analysis.add(an2)
            elif only_analyses and (an.name in only_analyses):
                logger.warning(f"Analysis '{an.name}' could not be performed for sample '{samplename}' because "\
                               f"not all required measurement groups or analyses were available. "
                               f"Skipping this analysis.")
    if len(needs_analysis):
        with (returner_context(conn) if conn is not None else get_engine_rw().begin()) as conn:
            # even though already evaluated analyses, still needs to go in order because anlsid's are acquired on upload
            for an in DDEF().get_higher_analyses_topo_sorted():
                if an not in needs_analysis: continue
                an_data: pd.DataFrame = data_by_mgoa[an.name] # type: ignore
                mgoa_to_loadanlsid[an.name]=\
                    upload_analysis(an, sample_info, an_data, 
                                pre_obtained_mgoa_to_loadanlsid=mgoa_to_loadanlsid, conn=conn)

                                