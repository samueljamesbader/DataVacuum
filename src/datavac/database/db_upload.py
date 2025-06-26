from __future__ import annotations

from datetime import datetime
from itertools import groupby
from typing import TYPE_CHECKING, Any, Mapping, Optional
from datavac.database.db_connect import get_engine_rw
from datavac.database.db_util import namewsq
from datavac.database.postgresql_upload_utils import upload_binary, upload_csv
from datavac.trove import Trove
from datavac.util.util import asnamedict, only, returner_context
from datavac.util.logging import logger
from sqlalchemy import Connection, delete, literal, select, text
from sqlalchemy.dialects.postgresql import insert as pgsql_insert, BYTEA, TIMESTAMP

from datavac.config.project_config import PCONF
from datavac.database.db_structure import DBSTRUCT

if TYPE_CHECKING:
    import pandas as pd
    from datavac.io.measurement_table import MeasurementTable

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

def reloads_to_rextracts(trove: Trove, sample_info: Mapping[str,Any], loadid: int,
                         conn: Connection, only_meas_groups: Optional[list[str]] = None):
    """ Move the Reload entries for a sample into the ReExtract table.
    Args:
        trove: The Trove for which the reloads should be moved.
        sample_info: A dictionary about the sample, keyed by DataDefinition.sample_info_columns
            Note that only the sample identifier column is used to identify the sample. The rest 
            of the sample_info is ignored.
        loadid: The loadid with which the reloads were addressed, which will now indicate
            the loadid that needs to be extracted.
        conn: The database connection to use.
        only_meas_groups: If provided, only reloads for these measurement groups will be moved
            to the ReExtract table. If None, all reloads for the sample within this trove will be moved.
    """
    samplename_col=PCONF().data_definition.SAMPLE_COLNAME
    sampletab=DBSTRUCT().get_sample_dbtable()
    relotab=DBSTRUCT().get_trove_dbtables(trove.name)['reload']
    reextab=DBSTRUCT().get_trove_dbtables(trove.name)['reextr']

    # Create a CTE by deleting the reloads for the sample
    statement=delete(relotab) \
                .where(sampletab.c[samplename_col]==sample_info[samplename_col]) \
                .where(sampletab.c.sampleid==relotab.c.sampleid)\
                .returning(relotab.c.MeasGroup)
    # If only_meas_groups is provided, restrict the deletion to those measurement groups
    if only_meas_groups is not None:
        statement=statement.where(relotab.c.MeasGroup.in_(only_meas_groups))
    anon=statement.cte()
    # Copy those deleted rows into the ReExtract table
    statement=pgsql_insert(reextab).from_select(['loadid'],select(literal(loadid)).select_from(anon))
    # Execute it all
    conn.execute(statement)
        
def upload_measurement(trove: Trove, sampleload_info: Mapping[str,Any], data_by_meas_group: Mapping[str, MeasurementTable],
                       only_meas_groups: Optional[list[str]] = None):
    sampletab=DBSTRUCT().get_sample_dbtable()
    loadtab=DBSTRUCT().get_trove_dbtables(trove.name)['loads']
    sample_info={k:v for k,v in sampleload_info.items()
                 if k in PCONF().data_definition.ALL_SAMPLE_COLNAMES}
    load_info={k:v for k,v in sampleload_info.items() if k in PCONF().data_definition.ALL_LOAD_COLNAMES(trove.name)}

    with get_engine_rw().begin() as conn:

        # Ensure sample exists
        sampleid = enter_sample(conn, **sample_info)

        # TODO: If there are prior loads for meas groups that *depend on* the ones being added,
        # or that the ones being added *depend on*, but those are not in the list of only_meas_groups,
        # raise an error

        # Delete prior loads for this sample (moving to ReLoad table)
        delete_prior_loads(trove, sample_info, conn, only_meas_groups=only_meas_groups)

    # Can commit here to release lock on sample table, for better scaling to many users,
    # but then should use a select-from in the loadid entry to get the sampleid again
    #with get_engine_rw().begin() as conn:
        for mg_name, meas_data in data_by_meas_group.items():

            # Put an entry into the Loads table and get the loadid
            loadid=conn.execute(pgsql_insert(loadtab)\
                               .values(sampleid=sampleid,MeasGroup=mg_name,**load_info)\
                               .returning(loadtab.c.loadid))\
                        .all()[0][0]
            
            # Upload the measurement data
            meastab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['meas']
            content=meas_data.scalar_table.reset_index()\
                           .assign(loadid=loadid).rename(columns={'index':'measid'})
            for ssr in PCONF().data_definition.subsample_references.values():
                ssr.transform(content, sample_info=sample_info, conn=conn)
            upload_csv(content[list(meastab.c.keys())],
                       conn, DBSTRUCT().int_schema, meastab.name)
            
            # Upload the sweeps
            sstab=meas_data.get_stacked_sweeps()
            if len(sstab):
                # This part is to handle the deprecated ONESTRING
                from datavac.database.postgresql_binary_format import pd_to_pg_converters
                sweepconv=(pd_to_pg_converters['STRING']) \
                    if (hasattr(mg_name,'ONESTRING') and mg_name.ONESTRING) else lambda s: s.tobytes() # type: ignore
                upload_binary(
                    sstab.assign(loadid=loadid)[['loadid','measid','sweep','header']],
                    conn,DBSTRUCT().int_schema,DBSTRUCT().get_measurement_group_dbtables(mg_name)['sweep'].name,
                    override_converters={'sweep':sweepconv,'header':pd_to_pg_converters['STRING']}
                )
    
    # Now since we've successfully uploaded the measurement, move the Reload entries to ReExtract
    # with get_engine_rw().begin() as conn:
        reloads_to_rextracts(trove, sample_info, loadid, conn, only_meas_groups=only_meas_groups)

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
    from datavac.database.db_structure import DBSTRUCT
    from datavac.util.logging import logger
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
    

def upload_extraction(trove: Trove, sampleload_info: Mapping[str,Any],
                      data_by_meas_group: Mapping[str, MeasurementTable],
                      only_meas_groups: Optional[list[str]] = None,
                      skip_delete_use_these_loadids: Optional[dict[str,list[int]]] = None):
    
    if skip_delete_use_these_loadids is not None:
        mg_name_to_loadid=skip_delete_use_these_loadids
    else:
        mg_name_to_loadid={k:only(v) for k,v in
           delete_prior_extractions(trove, sampleload_info, only_meas_groups=only_meas_groups).items()}

    for mg_name, meas_data in data_by_meas_group.items():
        
        # Upload the extraction data
        extrtab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['extr']
        content=meas_data.scalar_table.reset_index()\
                       .assign(loadid=mg_name_to_loadid[mg_name]).rename(columns={'index':'measid'})
        with get_engine_rw().begin() as conn:
            upload_csv(content[list(extrtab.c.keys())],
                       conn, DBSTRUCT().int_schema, extrtab.name)
        
            reextab=DBSTRUCT().get_trove_dbtables(trove.name)['reextr']
            statements=[
                delete(reextab).where(reextab.c.loadid==mg_name_to_loadid[mg_name]),
                # TODO: Add to reanalyze table
            ]
            conn.execute(text(';'.join([str(s.compile(conn,compile_kwargs={'literal_binds':True})) for s in statements])))
        
        

        

def upload_sample_descriptor(sd_name:str, data: pd.DataFrame, conn: Optional[Connection] = None):
    """ Upload a sample descriptor to the database.

    Args:
        sd_name: The name of the sample descriptor.
        data: A DataFrame containing the sample descriptor data.
    """
    # TODO: Inefficient, should be done in O(1) queries
    import pandas as pd
    from datavac.database.db_structure import DBSTRUCT
    from datavac.database.postgresql_upload_utils import upload_csv
    sampletab=DBSTRUCT().get_sample_dbtable()
    sdtab=DBSTRUCT().get_sample_descriptor_dbtable(sd_name)
    samplename_col=PCONF().data_definition.sample_identifier_column.name
    with (returner_context(conn) if conn else get_engine_rw().begin()) as conn:
        res=conn.execute(select(sampletab.c.sampleid,sampletab.c[samplename_col])\
                            .where(sampletab.c[samplename_col].in_(list(data.index)))).all()
        for samplename, row in data.iterrows():
            if samplename not in [r[1] for r in res]:
                enter_sample(conn, **PCONF().data_definition.sample_info_completer(
                    dict(**{samplename_col:samplename},**{k:v for k,v in row.items() if k in sampletab.c}))) # type: ignore
        res=conn.execute(select(sampletab.c.sampleid,sampletab.c[samplename_col])\
                            .where(sampletab.c[samplename_col].in_(list(data.index)))).all()
        data=pd.merge(left=data,right=pd.DataFrame(res,columns=['sampleid',samplename_col]),
                 how='left',left_index=True,right_on=samplename_col).drop(columns=[samplename_col]).set_index('sampleid')
        #print(data)
        conn.execute(delete(sdtab).where(sdtab.c.sampleid.in_(list(data.index))))
        upload_csv(data.reset_index(), conn, DBSTRUCT().int_schema, sd_name,)


def upload_subsample_reference(ssr_name: str, data: pd.DataFrame, conn: Optional[Connection] = None,
                               dump_extractions_and_analyses: bool = False):
    """ Upload a subsample reference to the database.

    Args:
        ssr_name: The name of the subsample reference.
        data: A DataFrame containing the subsample reference data.
    """
    from datavac.database.db_structure import DBSTRUCT
    from datavac.database.postgresql_upload_utils import upload_csv
    from datavac.config.project_config import PCONF
    from datavac.database.db_structure import DBSTRUCT
    from datavac.config.data_definition import DDEF
    from sqlalchemy import text
    from datavac.database.db_connect import get_engine_rw
    ssr = PCONF().data_definition.subsample_references[ssr_name]
    ssrtab = DBSTRUCT().get_subsample_reference_dbtable(ssr_name)
    with (returner_context(conn) if conn else get_engine_rw().begin()) as conn:

        # Load the data into a temporary table and check if it is different
        conn.execute(text(f'CREATE TEMP TABLE tmplay (LIKE {namewsq(ssrtab)});'))
        upload_csv(data, conn, None, 'tmplay')
        if conn.execute(text(
            f'''SELECT CASE WHEN EXISTS (TABLE {namewsq(ssrtab)} EXCEPT TABLE tmplay)
              OR EXISTS (TABLE tmplay EXCEPT TABLE {namewsq(ssrtab)})
            THEN 'different' ELSE 'same' END AS result ;''')).all()[0][0] == 'same':
            logger.debug(f"Content unchanged for {ssr_name}")
            
        # If the content has changed, we need to update the table
        else:
            logger.debug(f"Content changed for {ssr_name}, updating")

            # Remove foreign key constraints and delete old data if necessary
            for mg in DDEF().measurement_groups.values():
                if ssr_name in mg.subsample_reference_names:
                    meastab=DBSTRUCT().get_measurement_group_dbtables(mg.name)['meas']
                    if dump_extractions_and_analyses:
                        raise NotImplementedError()
                    conn.execute(text(f'ALTER TABLE {namewsq(meastab)}'\
                                      f' DROP CONSTRAINT IF EXISTS "fk_{ssr.key_column.name} -- {mg.name}";'))
            for an in DDEF().higher_analyses.values():
                if ssr_name in an.subsample_reference_names:
                    antab=DBSTRUCT().get_higher_analysis_dbtable(an.name)
                    if dump_extractions_and_analyses:
                        raise NotImplementedError()
                    conn.execute(text(f'ALTER TABLE {namewsq(antab)}'\
                                      f' DROP CONSTRAINT IF EXISTS "fk_{ssr.key_column.name} -- {an.name}";'))
                    
            # Bring in the new table
            conn.execute(delete(ssrtab))
            conn.execute(text(f'INSERT INTO {namewsq(ssrtab)} SELECT * from tmplay;'))
            
            # Recreate foreign key constraints
            for mg in DDEF().measurement_groups.values():
                if ssr_name in mg.subsample_reference_names:
                    meastab=DBSTRUCT().get_measurement_group_dbtables(mg.name)['meas']
                    conn.execute(text(f'ALTER TABLE {namewsq(meastab)}' \
                                      f' ADD CONSTRAINT "fk_{ssr.key_column.name} -- {mg.name}" FOREIGN KEY ("{ssr.key_column.name}")' \
                                      f' REFERENCES {namewsq(ssrtab)} ("{ssr.key_column.name}") ON DELETE CASCADE;'))
                    from datavac.database.db_create import create_meas_group_view
                    create_meas_group_view(mg.name, conn)
            for an in DDEF().higher_analyses.values():
                if ssr_name in an.subsample_reference_names:
                    antab=DBSTRUCT().get_higher_analysis_dbtable(an.name)
                    conn.execute(text(f'ALTER TABLE {namewsq(antab)}' \
                                      f' ADD CONSTRAINT "fk_{ssr.key_column.name} -- {an.name}" FOREIGN KEY ("{ssr.key_column.name}")' \
                                      f' REFERENCES {namewsq(ssrtab)} ("{ssr.key_column.name}") ON DELETE CASCADE;'))
                    from datavac.database.db_create import create_analysis_view
                    create_analysis_view(mg.name, conn)
                    
        conn.execute(text(f'DROP TABLE tmplay;'))
        #conn.commit()