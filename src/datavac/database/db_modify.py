from typing import Optional
from datavac.config.project_config import PCONF
from datavac.database.db_connect import get_engine_so
from datavac.database.db_structure import DBSTRUCT
from datavac.database.db_util import namews
from datavac.util.logging import logger
from datavac.util.util import only
from sqlalchemy import Connection, MetaData, delete, literal, select, Table
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
def dump_measurements(mg_name:str, conn:Connection):
    trove_name= PCONF().data_definition.measurement_groups[mg_name].trove_name()
    meas_tab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['meas']
    load_tab=DBSTRUCT().get_trove_dbtables(trove_name)['loads']
    reload_tab=DBSTRUCT().get_trove_dbtables(trove_name)['reload']
    assert [c.name for c in reload_tab.columns][:2]==['sampleid','MeasGroup'] # assumed by below SQL
    conn.execute(
        pgsql_insert(reload_tab) \
            .from_select([c.name for c in reload_tab.columns],
                         select(load_tab.c.sampleid,literal(mg_name),*[load_tab.c[c.name] for c in reload_tab.columns[2:]]) \
                         .select_from(meas_tab.join(load_tab)) \
                         .distinct()) \
            .on_conflict_do_nothing())
    conn.execute(delete(meas_tab))

def dump_extractions(mg_name:str, conn:Connection):
    trove_name= PCONF().data_definition.measurement_groups[mg_name].trove_name()
    meas_tab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['meas']
    extr_tab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['extr']
    reextr_tab=DBSTRUCT().get_trove_dbtables(trove_name)['reextr']
    conn.execute(
        pgsql_insert(reextr_tab) \
            .from_select([c.name for c in reextr_tab.columns],
                         select(meas_tab.c.loadid,literal(mg_name)) \
                         .select_from(meas_tab) \
                         .distinct()) \
            .on_conflict_do_nothing())
    conn.execute(delete(extr_tab))
    

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
                    dump_measurements(mg_name, conn)
                    drops=[db_metadata.tables[namews(t)]
                               for t in desired_tabs.values()
                                    if namews(t) in db_metadata.tables]
                    db_metadata.drop_all(conn,drops,checkfirst=True)
                    for drop in drops: db_metadata.remove(drop)
                else: need_to_create_meas = False
            else: need_to_create_meas = True
            if need_to_create_meas: desired_tabs['meas'].create(conn)

            if namews(desired_tabs['extr']) in db_metadata.tables:
                if force_extr or _table_mismatch(desired_tabs['extr'], db_metadata):
                    need_to_create_extr = True
                    dump_extractions(mg_name, conn)
                    db_metadata.tables[namews(desired_tabs['extr'])].drop(conn,checkfirst=True)
                else: need_to_create_extr = False
            else: need_to_create_extr = True
            if need_to_create_extr: desired_tabs['extr'].create(conn)

            if namews(desired_tabs['sweep']) not in db_metadata.tables:
                desired_tabs['sweep'].create(conn)

            



#def force_database():
#    db_metadata = MetaData(schema=DBSTRUCT().int_schema)
#    db_metadata.reflect(bind=get_engine_so(), views=True)
#    #print(db_metadata.tables)
#
#    with get_engine_so().begin() as conn:
#        _setup_foundation(conn)
#
#    for sr in PCONF().data_definition.sample_references:
#        sr_tab=DBSTRUCT().get_sample_reference_dbtable(sr)
#        if sr_tab.name in db_metadata.tables:
#
#
#
#
#    for mg in PCONF().data_definition.measurement_groups:
#        DBSTRUCT().get_measurement_group_dbtables(mg)
#        