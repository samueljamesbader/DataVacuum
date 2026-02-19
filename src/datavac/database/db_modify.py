from functools import reduce
from typing import Any, Optional
from datavac.config.project_config import PCONF
from datavac.database.db_connect import get_engine_rw, get_engine_so
from datavac.database.db_structure import DBSTRUCT
from datavac.config.data_definition import DDEF, HigherAnalysis
from datavac.database.db_util import namews
from datavac.util.dvlogging import logger
from datavac.util.util import only, returner_context
from sqlalchemy import INTEGER, Column, Connection, MetaData, delete, literal, select, Table, text, values
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
    all_desired_tab_names = set(tab.name for tab in all_desired_tabs)
    with get_engine_so().begin() as conn:
        db_metadata = MetaData(schema=DBSTRUCT().int_schema)
        db_metadata.reflect(bind=conn, only=(lambda tn,md:(tn in all_desired_tab_names)), views=True)

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
def update_analysis_tables(specific_analyses:Optional[list[str]]=None, force=False):
    from datavac.database.db_upload_meas import delete_prior_analyses
    if specific_analyses is None:
        specific_analyses = list(DDEF().higher_analyses)
    all_desired_tabs = [tab for an in specific_analyses for tab in DBSTRUCT().get_higher_analysis_dbtables(an).values()]
    all_desired_tab_names = set(tab.name for tab in all_desired_tabs)
    with get_engine_so().begin() as conn:
        db_metadata = MetaData(schema=DBSTRUCT().int_schema)
        db_metadata.reflect(bind=conn, only=(lambda tn,md:(tn in all_desired_tab_names)), views=True)
        for an_name in specific_analyses:
            desired_tabs = DBSTRUCT().get_higher_analysis_dbtables(an_name)
            if namews(desired_tabs['aidt']) in db_metadata.tables:
                if force or _table_mismatch(desired_tabs['aidt'], db_metadata):
                    delete_prior_analyses(None, only_analyses=[an_name], conn=conn)
                    need_to_create_aidt = True
                    conn.execute(text(f"""DROP VIEW IF EXISTS {DBSTRUCT().jmp_schema}."{an_name}" """))
                    drops=[db_metadata.tables[namews(t)]
                               for t in desired_tabs.values()
                                    if namews(t) in db_metadata.tables]
                    db_metadata.drop_all(conn,drops,checkfirst=True)
                    for drop in drops: db_metadata.remove(drop)
                else: need_to_create_aidt = False
            else: need_to_create_aidt = True
            if need_to_create_aidt: desired_tabs['aidt'].create(conn)
            if namews(desired_tabs['anls']) in db_metadata.tables:
                if force or _table_mismatch(desired_tabs['anls'], db_metadata):
                    if not need_to_create_aidt:
                        delete_prior_analyses(None, only_analyses=[an_name], conn=conn)
                    need_to_create_anls = True
                    conn.execute(text(f"""DROP VIEW IF EXISTS {DBSTRUCT().jmp_schema}."{an_name}" """))
                    db_metadata.tables[namews(desired_tabs['anls'])].drop(conn,checkfirst=True)
                    db_metadata.remove(db_metadata.tables[namews(desired_tabs['anls'])])
                else: need_to_create_anls = False
            else: need_to_create_anls = True
            if need_to_create_anls: desired_tabs['anls'].create(conn)
            if need_to_create_aidt or need_to_create_anls:
                from datavac.database.db_create import create_analysis_view
                create_analysis_view(an_name, conn)


def heal(trove_names:Optional[list[str]]=None, only_samples:Optional[list[str]]=None):
    import pandas as pd
    from datavac.database.db_upload_meas import read_and_enter_data
    from datavac.database.db_upload_meas import perform_and_enter_extraction, perform_and_enter_analysis
    from datavac.config.data_definition import DDEF
    sampletab=DBSTRUCT().get_sample_dbtable()
    trove_names= trove_names if trove_names is not None else list(PCONF().data_definition.troves.keys())
    for trove in (v for k,v in PCONF().data_definition.troves.items() if k in trove_names):
        load_grouping = trove.natural_grouping or DDEF().SAMPLE_COLNAME

        if len(PCONF().data_definition.troves)>1:
            logger.debug(f"Healing loads for trove {trove.name}")
        relotab=DBSTRUCT().get_trove_dbtables(trove_name=trove.name)['reload']
        with get_engine_rw().begin() as conn:
            sql_relo=select(*relotab.c, *sampletab.c).select_from(relotab.join(sampletab))
            if only_samples is not None:
                sql_relo=sql_relo.where(sampletab.c[DDEF().SAMPLE_COLNAME].in_(only_samples))
            pd_relo=pd.read_sql(con=conn, sql=sql_relo)
        if not len(pd_relo): logger.debug(f"No reloads required")
        else:
            for lgval, grp in pd_relo.groupby(load_grouping):
                meas_groups = grp['MeasGroup'].unique().tolist()
                logger.debug(f"Healing for {load_grouping}={lgval}, measurement groups {meas_groups}")
                if load_grouping == DDEF().SAMPLE_COLNAME:
                    sampleload_info: dict[str,Any] = {load_grouping: [lgval]}
                else:
                    sampleload_info: dict[str,Any] = {load_grouping: [lgval], DDEF().SAMPLE_COLNAME: grp[DDEF().SAMPLE_COLNAME].unique().tolist()}
                read_and_enter_data(trove_names=[trove.name],only_meas_groups=meas_groups,only_sampleload_info=sampleload_info)
        logger.debug(f"Done with reloads")


        reextab=DBSTRUCT().get_trove_dbtables(trove_name=trove.name)['reextr']
        loadtab=DBSTRUCT().get_trove_dbtables(trove_name=trove.name)['loads']
        with get_engine_rw().begin() as conn:
            sql_reex=sql=select(*reextab.c, *loadtab.c, *sampletab.c)\
                           .select_from(reextab.join(loadtab).join(sampletab))
            if only_samples is not None:
                sql_reex=sql_reex.where(sampletab.c[DDEF().SAMPLE_COLNAME].in_(only_samples))
            pd_reex= pd.read_sql(con=conn, sql=sql_reex)
        if not len(pd_reex): logger.debug(f"No re-extractions required")
        else:
            for samplename, grp in pd_reex.groupby(DDEF().SAMPLE_COLNAME):
                sample_info = {c.name: only(grp[c.name].unique()) for c in sampletab.c if c.name!='sampleid'}
                with get_engine_rw().begin() as conn:
                    data_by_mg,mg_to_loadid=perform_and_enter_extraction(trove=trove, samplename=samplename,
                        only_meas_groups=grp['MeasGroup'].unique().tolist(),conn=conn)
                    needed_analyses = list(set(an.name for mg in data_by_mg
                                              for an in DDEF().get_meas_groups_dependent_graph()[DDEF().measurement_groups[mg]]
                                                if isinstance(an, HigherAnalysis)))
                    perform_and_enter_analysis(sample_info=sample_info, only_analyses=needed_analyses, conn=conn,
                                               pre_obtained_data_by_mgoa=data_by_mg, pre_obtained_mgoa_to_loadanlsid=mg_to_loadid,)
            
        reantab=DBSTRUCT().get_higher_analysis_reload_table()
        with get_engine_rw().begin() as conn:
            sql_rean=select(*reantab.c,*sampletab.c)\
               .select_from(reantab.join(sampletab))
            if only_samples is not None:
                sql_rean=sql_rean.where(sampletab.c[DDEF().SAMPLE_COLNAME].in_(only_samples))
            pd_rean= pd.read_sql(con=conn, sql=sql_rean)
        if not len(pd_rean): logger.debug(f"No re-analyses required")
        else:
            for samplename, grp in pd_rean.groupby(DDEF().SAMPLE_COLNAME):
                logger.debug(f"Re-analyzing {samplename} for {grp['Analysis'].unique()}")
                sample_info = {c.name: only(grp[c.name].unique()) for c in sampletab.c if c.name!='sampleid'}
                analyses = grp['Analysis'].unique().tolist()
                perform_and_enter_analysis(sample_info=sample_info, only_analyses=analyses)
            
def run_new_analysis(an_name: str):
    """Runs a new analysis on all data in the database."""
    import pandas as pd
    from datavac.database.db_upload_meas import perform_and_enter_analysis
    from datavac.database.db_create import create_analysis_view
    from datavac.database.db_connect import get_engine_rw

    an= DDEF().higher_analyses[an_name]
    sels=[]
    if len(an.required_dependencies):
        for mgoa_name in an.required_dependencies:
            if mgoa_name in DDEF().measurement_groups:
                trove= DDEF().measurement_groups[mgoa_name].trove()
                loadtab=trove.dbtables('loads')
                sels.append(select(loadtab.c.sampleid).where(loadtab.c.MeasGroup==literal(mgoa_name)))
            elif mgoa_name in DDEF().higher_analyses:
                an2 = DDEF().higher_analyses[mgoa_name]
                sels.append(select(an2.dbtables('aidt').c.sampleid))
            else:
                raise ValueError(f"Unknown measurement group or analysis {mgoa_name} in {an_name}")
        from sqlalchemy import intersect
        sampleid_subquery = intersect(*sels)
    else:
        for mgoa_name in an.optional_dependencies:
            if mgoa_name in DDEF().measurement_groups:
                trove= DDEF().measurement_groups[mgoa_name].trove()
                loadtab=trove.dbtables('loads')
                sels.append(select(loadtab.c.sampleid).where(loadtab.c.MeasGroup==literal(mgoa_name)))
            elif mgoa_name in DDEF().higher_analyses:
                an2 = DDEF().higher_analyses[mgoa_name]
                sels.append(select(an2.dbtables('aidt').c.sampleid))
            else:
                raise ValueError(f"Unknown measurement group or analysis {mgoa_name} in {an_name}")
        from sqlalchemy import union
        sampleid_subquery = union(*sels)
        
    
    sampletab=DBSTRUCT().get_sample_dbtable()
    query=select(*sampletab.c).select_from(
                            sampletab.join(sampleid_subquery, # type: ignore
                                           sampletab.c.sampleid==sampleid_subquery.c.sampleid))
    with get_engine_rw().connect() as conn:
        sample_infos = pd.read_sql(query, conn)

        reantab=DBSTRUCT().get_higher_analysis_reload_table()
        sampleid_cte=select(
            values(Column('sampleid', INTEGER),name="needed_sampleids").data(
                [(s,) for s in sample_infos['sampleid']])).cte()
        conn.execute(pgsql_insert(reantab).from_select(['sampleid','Analysis'],
                                              select(sampleid_cte.c.sampleid,literal(an_name)))\
                             .on_conflict_do_nothing())
    with get_engine_rw().begin() as conn:
        for sampleinfo in sample_infos.to_dict(orient='records'):
            perform_and_enter_analysis(sample_info=sampleinfo, only_analyses=[an_name], conn=conn) # type: ignore