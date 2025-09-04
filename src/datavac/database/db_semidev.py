from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Optional, cast

from datavac.appserve.api import client_server_split
from datavac.database.db_util import namews
from datavac.util.dvlogging import logger
from datavac.util.util import returner_context

if TYPE_CHECKING:
    from sqlalchemy import Connection
    from pandas import DataFrame

# TODO: This function is ported from old framework
def upload_mask_info(mask_info: dict[str, Any],conn: Optional[Connection]=None):
    from sqlalchemy.dialects.postgresql import insert as pgsql_insert
    from sqlalchemy import select
    import pandas as pd
    from datavac.util.util import import_modfunc
    from datavac.database.db_structure import DBSTRUCT
    from datavac.database.postgresql_upload_utils import upload_csv
    from datavac.database.db_connect import get_engine_rw
    
    with (returner_context(conn) if conn else get_engine_rw().begin()) as conn:
        masktab = DBSTRUCT().get_sample_reference_dbtable('MaskSet')
        diemtab = DBSTRUCT().get_subsample_reference_dbtable('Dies')
        if not len(mask_info): return
        diemdf=[]
        for mask,info in mask_info.items():
            #dbdf,to_pickle=import_modfunc(info['generator'])(**info['args'])
            dbdf, to_pickle = info
            diemdf.append(dbdf.assign(MaskSet=mask)[[c.name for c in diemtab.columns if c.name!='dieid']])
            update_info=dict(MaskSet=mask,info_pickle=pickle.dumps(to_pickle))
            conn.execute(pgsql_insert(masktab).values(**update_info)\
                         .on_conflict_do_update(index_elements=['MaskSet'],set_=update_info))

        diemdf=pd.concat(diemdf).reset_index(drop=True).reset_index(drop=False)
        previous_dietab=pd.read_sql(select(*diemtab.columns).order_by(diemtab.c['dieid']),conn).reset_index(drop=False)
        # This checks that nothing has changed in the previous table
        # very important to check that because all the measured data is only associated with a die index,
        # so if we accidentally change the die index, even by uploading the tables in a different order...
        # poof all the old data is now associated with the wrong dies or even wrong masks!!
        #print("\n\n\nPREVIOUS:")
        #print(previous_dietab)
        #print("\n\n\nNEW:")
        #print(diemdf)
        #print("\n")
        assert len(previous_dietab.merge(diemdf))==len(previous_dietab),\
            "Can't add to die tables without messing up existing dies"
        upload_csv(diemdf.iloc[len(previous_dietab):].rename(columns={'index':'dieid'}),conn,DBSTRUCT().int_schema,'Dies')

        #print("Successful")


def _update_layout_param_group(layout_param_group: str, conn: Optional[Connection], dump_extractions_and_analyses: bool = True):
    from datavac.database.db_upload_other import upload_subsample_reference
    from datavac.config.data_definition import DDEF, SemiDeviceDataDefinition
    upload_subsample_reference(f'LayoutParams -- {layout_param_group}',
           cast(SemiDeviceDataDefinition,DDEF()).get_layout_params_table(layout_param_group).reset_index(drop=False),
           conn=conn,dump_extractions_and_analyses=dump_extractions_and_analyses)
    
def update_layout_params(conn: Optional[Connection] = None, dump_extractions_and_analyses: bool = True):
    from datavac.config.data_definition import DDEF, SemiDeviceDataDefinition
    from datavac.database.db_connect import get_engine_so
    from datavac.config.layout_params import LP
    LP(force_regenerate=True) # Regenerate the layout parameters
    lpnames=cast(SemiDeviceDataDefinition,DDEF()).get_layout_params_table_names()
    with (returner_context(conn) if conn else get_engine_so().begin()) as conn:
        for layout_param_group in lpnames:
            _update_layout_param_group(layout_param_group, conn,
                                       dump_extractions_and_analyses=dump_extractions_and_analyses)

def upload_splits(specific_splits: Optional[list[str]]=None, conn: Optional[Connection] = None):
    from datavac.config.data_definition import DDEF, SemiDeviceDataDefinition
    from datavac.database.db_connect import get_engine_rw
    from datavac.database.db_upload_other import upload_sample_descriptor
    from datavac.config.sample_splits import get_flow_names, get_split_table
    split_manager = cast(SemiDeviceDataDefinition, DDEF()).split_manager
    with (returner_context(conn) if conn else get_engine_rw().begin()) as conn:
        for flow_name in get_flow_names(force_external=True):
            if specific_splits is not None and flow_name not in specific_splits:
                continue
            split_table = get_split_table(flow_name, force_external=True)
            assert len(split_table)
            logger.info(f"Uploading split table for flow {flow_name} with {len(split_table)} rows")
            print(split_table)
            upload_sample_descriptor(f'SplitTable -- {flow_name}', split_table.set_index(DDEF().SAMPLE_COLNAME),
                                     conn=conn, clear_all_previous=True)
            
def create_split_table_view(sp_name: str, conn: Optional[Connection]=None, just_DDL_string: bool = False) -> str:
    from datavac.config.data_definition import DDEF
    from datavac.database.db_structure import DBSTRUCT
    from sqlalchemy import select, text
    sp_tab=DBSTRUCT().get_sample_descriptor_dbtable(f"SplitTable -- {sp_name}")
    sample_tab=DBSTRUCT().get_sample_dbtable()
    sel=select(*[c for c in sample_tab.c if c.name!='sampleid'],
               *[c for c in sp_tab.c if c.name!='sampleid']).select_from(
        sp_tab.join(sample_tab,sp_tab.c.sampleid==sample_tab.c.sampleid)
    ).order_by(sp_tab.c.sampleid)
    seltextl=sel.compile(conn, compile_kwargs={"literal_binds": True})
    view_namewsq = f'{DBSTRUCT().jmp_schema}."{sp_name}"'
    ddl=f"""CREATE OR REPLACE VIEW {view_namewsq} AS {seltextl}"""
    if not just_DDL_string:
        assert conn is not None, "Connection must be provided if not just generating DDL string."
        # sel.compile converts % in column names to %%
        # https://github.com/sqlalchemy/sqlalchemy/discussions/8077
        print("CREATING SPLIT TABLE VIEW",ddl)
        conn.execute(text(ddl.replace("%%","%"))) 
    return ddl

def update_split_tables(specific_splits:Optional[list[str]]=None, force=False, conn: Optional[Connection] = None):
    from datavac.config.data_definition import SemiDeviceDataDefinition, DDEF
    from datavac.config.sample_splits import get_flow_names
    from datavac.database.db_structure import DBSTRUCT
    from datavac.database.db_modify import _table_mismatch
    from datavac.database.db_connect import get_engine_so
    from datavac.database.db_connect import avoid_db_if_possible
    from sqlalchemy import MetaData
    from sqlalchemy import text
    ddef = cast(SemiDeviceDataDefinition, DDEF())
    split_manager= ddef.split_manager
    with avoid_db_if_possible():
        if specific_splits is None:
            specific_splits = list(get_flow_names(force_external=True))
        all_desired_tabs = [DBSTRUCT().get_sample_descriptor_dbtable(f'SplitTable -- {sp}') for sp in specific_splits]
    with (returner_context(conn) if conn else get_engine_so().begin()) as conn:
        db_metadata = MetaData(schema=DBSTRUCT().int_schema)
        db_metadata.reflect(bind=conn, only=[tab.name for tab in all_desired_tabs], views=True)
        for sp_name in specific_splits:
            with avoid_db_if_possible():
                desired_tab = DBSTRUCT().get_sample_descriptor_dbtable(f'SplitTable -- {sp_name}')
            if namews(desired_tab) in db_metadata.tables:
                if force or _table_mismatch(desired_tab, db_metadata):
                    need_to_create = True
                    conn.execute(text(f"""DROP VIEW IF EXISTS {DBSTRUCT().jmp_schema}."{sp_name}" """))
                    drops=[desired_tab]
                    db_metadata.drop_all(conn,drops,checkfirst=True)
                    for drop in drops: db_metadata.remove(drop)
                else: need_to_create = False
            else: need_to_create = True
            if need_to_create: desired_tab.create(conn)
            if need_to_create:
                create_split_table_view(sp_name, conn)
            upload_splits(specific_splits=[sp_name], conn=conn)
