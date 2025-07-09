from typing import Any, Optional
from datavac.config.project_config import PCONF
from datavac.database.db_connect import get_engine_rw, get_engine_so
from datavac.database.db_structure import DBSTRUCT
from datavac.config.data_definition import DDEF, HigherAnalysis
from datavac.database.db_util import namews
from datavac.util.dvlogging import logger
from datavac.util.util import only
from sqlalchemy import Connection, MetaData, delete, literal, select, Table, text
from sqlalchemy.dialects.postgresql import insert as pgsql_insert

def _table_mismatch(described_table: Table, db_metadata: MetaData):
    existing_table = db_metadata.tables[namews(described_table)]

    if not [c.name for c in existing_table.columns]==[c.name for c in described_table.columns]:
        logger.warning(f"Column name (or ordering) mismatch in {described_table.name}")
        logger.warning(f"Currently in DB: {[c.name for c in existing_table.columns]}")
        logger.warning(f"Should be in DB: {[c.name for c in described_table.columns]}")
        return True
    if len(type_misses:={k:(v1,v2) for k,(v1,v2) in 
                                {c.name: (c.type.__class__, described_table.c[c.name].type.__class__)
                                    for c in existing_table.columns}.items()
                            if v1!=v2}):
        logger.warning(f"Column type mismatch in {described_table.name}")
        logger.warning(f"Currently in DB: { {k:v1 for k,(v1,v2) in type_misses.items()} }")
        logger.warning(f"Should be in DB: { {k:v2 for k,(v1,v2) in type_misses.items()} }")
        return True
    
#def dump_measurements(mg_name:str, conn:Connection):
#    trove_name= PCONF().data_definition.measurement_groups[mg_name].trove_name()
#    meas_tab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['meas']
#    load_tab=DBSTRUCT().get_trove_dbtables(trove_name)['loads']
#    reload_tab=DBSTRUCT().get_trove_dbtables(trove_name)['reload']
#    assert [c.name for c in reload_tab.columns][:2]==['sampleid','MeasGroup'] # assumed by below SQL
#    conn.execute(
#        pgsql_insert(reload_tab) \
#            .from_select([c.name for c in reload_tab.columns],
#                         select(load_tab.c.sampleid,literal(mg_name),*[load_tab.c[c.name] for c in reload_tab.columns[2:]]) \
#                         .select_from(meas_tab.join(load_tab)) \
#                         .distinct()) \
#            .on_conflict_do_nothing())
#    conn.execute(delete(meas_tab))
#
#def dump_extractions(mg_name:str, conn:Connection):
#    trove_name= PCONF().data_definition.measurement_groups[mg_name].trove_name()
#    meas_tab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['meas']
#    extr_tab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['extr']
#    reextr_tab=DBSTRUCT().get_trove_dbtables(trove_name)['reextr']
#    conn.execute(
#        pgsql_insert(reextr_tab) \
#            .from_select([c.name for c in reextr_tab.columns],
#                         select(meas_tab.c.loadid) \
#                         .select_from(meas_tab) \
#                         .distinct()) \
#            .on_conflict_do_nothing())
#    conn.execute(delete(extr_tab))
    

def update_measurement_group_tables(specific_groups:Optional[list[str]]=None, # type: ignore
                                    force_meas=False, force_extr=False):
    if specific_groups is None:
        specific_groups:list[str] = list(PCONF().data_definition.measurement_groups)

    all_desired_tabs = [tab for mg in specific_groups for tab in DBSTRUCT().get_measurement_group_dbtables(mg).values()]
    with get_engine_so().begin() as conn:
        db_metadata = MetaData(schema=DBSTRUCT().int_schema)
        db_metadata.reflect(bind=conn, only=[tab.name for tab in all_desired_tabs], views=True)

        for mg_name in specific_groups:
            desired_tabs= DBSTRUCT().get_measurement_group_dbtables(mg_name)
            if namews(desired_tabs['meas']) in db_metadata.tables:
                if force_meas or _table_mismatch(desired_tabs['meas'], db_metadata):
                    need_to_create_meas = True
                    from datavac.database.db_upload_meas import delete_prior_loads
                    delete_prior_loads(trove=DDEF().troves[only(DDEF().measurement_groups[mg_name].reader_cards)],
                                       sample_info=None,only_meas_groups=[mg_name], conn=conn)
                    drops=[db_metadata.tables[namews(t)]
                               for t in desired_tabs.values()
                                    if namews(t) in db_metadata.tables]
                    conn.execute(text(f"""DROP VIEW IF EXISTS {DBSTRUCT().jmp_schema}."{mg_name}" """))
                    db_metadata.drop_all(conn,drops,checkfirst=True)
                    for drop in drops: db_metadata.remove(drop)
                else: need_to_create_meas = False
            else: need_to_create_meas = True
            if need_to_create_meas: desired_tabs['meas'].create(conn)

            if namews(desired_tabs['extr']) in db_metadata.tables:
                if force_extr or _table_mismatch(desired_tabs['extr'], db_metadata):
                    need_to_create_extr = True
                    from datavac.database.db_upload_meas import delete_prior_extractions
                    delete_prior_extractions(trove=DDEF().troves[only(DDEF().measurement_groups[mg_name].reader_cards)],
                                       sampleload_info=None,only_meas_groups=[mg_name], conn=conn)
                    conn.execute(text(f"""DROP VIEW IF EXISTS {DBSTRUCT().jmp_schema}."{mg_name}" """))
                    db_metadata.tables[namews(desired_tabs['extr'])].drop(conn,checkfirst=True)
                else: need_to_create_extr = False
            else: need_to_create_extr = True
            if need_to_create_extr: desired_tabs['extr'].create(conn)

            if namews(desired_tabs['sweep']) not in db_metadata.tables:
                desired_tabs['sweep'].create(conn)

            if need_to_create_meas or need_to_create_extr:
                from datavac.database.db_create import create_meas_group_view
                create_meas_group_view(mg_name, conn)

def heal():
    import pandas as pd
    from datavac.database.db_upload_meas import read_and_enter_data
    from datavac.database.db_upload_meas import perform_and_enter_extraction, perform_and_enter_analysis
    sampletab=DBSTRUCT().get_sample_dbtable()
    for trove in PCONF().data_definition.troves.values():
        load_grouping = trove.natural_grouping or DDEF().SAMPLE_COLNAME

        if len(PCONF().data_definition.troves)>1:
            logger.debug(f"Healing loads for trove {trove.name}")
        relotab=DBSTRUCT().get_trove_dbtables(trove_name=trove.name)['reload']
        with get_engine_rw().begin() as conn:
            pd_relo=pd.read_sql(con=conn, sql=select(*relotab.c, *sampletab.c).select_from(relotab.join(sampletab)))
        if not len(pd_relo): logger.debug(f"No reloads required")
        else:
            for lgval, grp in pd_relo.groupby(load_grouping):
                meas_groups = list(grp['MeasGroup'].unique())
                logger.debug(f"Healing for {load_grouping}={lgval}, measurement groups {meas_groups}")
                if load_grouping == DDEF().SAMPLE_COLNAME:
                    sampleload_info: dict[str,Any] = {load_grouping: [lgval]}
                else:
                    sampleload_info: dict[str,Any] = {load_grouping: [lgval], DDEF().SAMPLE_COLNAME: list(grp['sampleid'].unique())}
                read_and_enter_data(only_meas_groups=meas_groups,only_sampleload_info=sampleload_info)
        logger.debug(f"Done with reloads")


        reextab=DBSTRUCT().get_trove_dbtables(trove_name=trove.name)['reextr']
        loadtab=DBSTRUCT().get_trove_dbtables(trove_name=trove.name)['loads']
        with get_engine_rw().begin() as conn:
            pd_reex= pd.read_sql(con=conn, sql=select(*reextab.c, *loadtab.c, *sampletab.c)\
               .select_from(reextab.join(loadtab).join(sampletab)))
        if not len(pd_reex): logger.debug(f"No re-extractions required")
        else:
            for samplename, grp in pd_reex.groupby(DDEF().SAMPLE_COLNAME):
                with get_engine_rw().begin() as conn:
                    data_by_mg,mg_to_loadid=perform_and_enter_extraction(trove=trove, samplename=samplename,
                        only_meas_groups=list(grp['MeasGroup'].unique()),conn=conn)
                    needed_analyses = list(set(an.name for mg in data_by_mg
                                              for an in DDEF().get_meas_groups_dependent_graph()[DDEF().measurement_groups[mg]]
                                                if isinstance(an, HigherAnalysis)))
                    perform_and_enter_analysis(samplename=samplename, only_analyses=needed_analyses, conn=conn,
                                               pre_obtained_data_by_mgoa=data_by_mg, pre_obtained_mgoa_to_loadanlsid=mg_to_loadid,)
            
        reantab=DBSTRUCT().get_higher_analysis_reload_table()
        with get_engine_rw().begin() as conn:
            pd_rean= pd.read_sql(con=conn, sql=select(*reantab.c,*sampletab.c)\
               .select_from(reantab.join(sampletab)))
        if not len(pd_rean): logger.debug(f"No re-analyses required")
        else:
            for samplename, grp in pd_rean.groupby(DDEF().SAMPLE_COLNAME):
                analyses = list(grp['Analysis'].unique())
                perform_and_enter_analysis(samplename=samplename, only_analyses=analyses)
            
            