import argparse
import functools
import os
import pickle
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
import time
from pathlib import Path
from typing import Union, Callable, Optional

import requests
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from datavac.appserve.dvsecrets import get_db_connection_info
from datavac.io.layout_params import get_layout_params
from datavac.io.meta_reader import ensure_meas_group_sufficiency, ALL_MATERIAL_COLUMNS, ALL_LOAD_COLUMNS, \
    ALL_MATLOAD_COLUMNS
from datavac.io.postgresql_binary_format import df_to_pgbin, pd_to_pg_converters
from sqlalchemy import text, Engine, create_engine, ForeignKey, UniqueConstraint, PrimaryKeyConstraint, \
    ForeignKeyConstraint, DOUBLE_PRECISION, delete, select, literal, union_all, insert, Connection, label, join
from sqlalchemy.dialects.postgresql import insert as pgsql_insert, BYTEA, TIMESTAMP
from sqlalchemy import INTEGER, VARCHAR, BOOLEAN, Column, Table, MetaData
import numpy as np
from sqlalchemy.engine import URL
import io
import pandas as pd

from datavac.util.cli import cli_helper
from datavac.util.conf import CONFIG
from datavac.util.logging import logger, time_it
from datavac.util.paths import USER_CACHE
from datavac.util.util import returner_context, import_modfunc

_CASC=dict(onupdate='CASCADE',ondelete='CASCADE')

_database:'PostgreSQLDatabase'=None
def get_database(metadata_source:Optional[str]=None,populate_metadata:bool=True) -> 'PostgreSQLDatabase':
    """ Returns a singleton PostgreSQLDatabase object

    Arguments allow determining the source of the metadata (local yaml or DB itself), and, in the case of local
    generation, allow postponing the metadata population.  Note: the Blob Store table will exist in metadata even
    if the metadata is not populated since that is generally required by routines which run during metadata creation.

    Args:
        metadata_source (str): If 'yaml', then generate the metadata (if needed) based on the local yaml.
            If 'reflect', then reflect the actual database into metadata.  If this parameter is supplied and
            the singleton already exists, it must match the metadata source of the singleton.  If not supplied,
            and creating the singleton, defaults to 'yaml'
        populate_metadata (bool): Default True.  If the metadata source is 'reflect', then this argument is ignored and
            the metadata will be reflected from the database.  If the metadata source is 'yaml', this argument can be
            set False to avoid building up the metadata from the yaml.  This is useful for some operations which need
            a database object but may be indirectly called during the process of building the metadata.
    Returns:
        a PostgreSQLDatabase singleton
    """
    global _database
    assert 'postgre' in get_db_connection_info()['Driver'], \
        "Only supported database driver is 'postgresql'."
    if metadata_source is not None: assert metadata_source in ['yaml','reflect']
    if not _database:
        _database=PostgreSQLDatabase(metadata_source=(metadata_source or 'yaml'))
    if metadata_source and (metadata_source!=_database._metadata_source):
        raise ValueError("Can't change metadata source after database is initialized")
    if populate_metadata and not _database._populated_metadata:
        _database.establish_database(just_metadata=True)
    return _database

def get_blob_store() -> 'PostgreSQLDatabase':
    return get_database(populate_metadata=False)

class Database:
    def get_data(self,meas_group,scalar_columns=None,include_sweeps=False,
                 unstack_headers=False,raw_only=False,**factors):
        raise NotImplementedError
    def get_factors(self,meas_group,factor_names,pre_filters={}):
        raise NotImplementedError

# TODO: Right now, a lot of PostgreSQL-specific functions are used by AlchemyDatabase
class AlchemyDatabase:

    engine:  Engine
    _metadata: MetaData

    def __init__(self, metadata_source='yaml'):
        self._populated_metadata=False
        self._metadata_source=metadata_source
        assert self._metadata_source in ['yaml','reflect']
        with time_it("Initializing Database",threshold_time=.1):
            with time_it("Make engine", threshold_time=.1):
                self._make_engine()
            with time_it("Init metadata", threshold_time=.1):
                self._init_metadata(reflect=(self._metadata_source=='reflect'))

    @contextmanager
    def engine_connect(self,*args,**kwargs):
        tc=time_it('Engine connect',threshold_time=.02)
        tc.__enter__()
        with self.engine.connect() as conn:
            tc.__exit__(None,None,None)
            yield conn
    @contextmanager
    def engine_begin(self,*args,**kwargs):
        tc=time_it('Engine begin',threshold_time=.02)
        tc.__enter__()
        with self.engine.begin() as conn:
            tc.__exit__(None,None,None)
            yield conn

    def validate(self, on_mismatch='raise'):
        assert self._metadata_source=='reflect', "Need to reflect DB metadata to validate it!"
        self.establish_database(on_mismatch=on_mismatch,just_metadata=False)

    def establish_database(self, on_mismatch='raise', just_metadata=True):
        raise NotImplementedError

    def _make_engine(self):
        connection_info=get_db_connection_info()
        url=URL.create(
            drivername=connection_info['Driver'],
            username=connection_info['Uid'],
            password=connection_info['Password'],
            host=connection_info['Server'],
            port=int(connection_info['Port']),
            database=connection_info['Database'],
        )
        self.engine=create_engine(url, connect_args=connection_info['sslargs'], pool_recycle=60)

    def establish_schema_and_blob_store(self, conn, on_mismatch='raise', just_metadata=False): raise NotImplementedError
    def get_obj(self,name): raise NotImplementedError
    def store_obj(self,name,obj,conn=None): raise NotImplementedError

    def _init_metadata(self, reflect=False):
        self._metadata = MetaData(schema=CONFIG['database']['schema_names']['internal'])
        if reflect:
            with time_it("Reflecting",threshold_time=.1):
                with self.engine_connect() as conn:
                    self._metadata.reflect(conn)
            self._populated_metadata=True
        else:
            self.establish_schema_and_blob_store(conn=None, just_metadata=True)

    def clear_database(self, only_tables=None, schema_also=False, conn=None):
        removed_tables=[]
        with (returner_context(conn) if conn else self.engine_begin()) as conn:
            if schema_also:
                for schema in CONFIG['database']['schema_names'].values():
                    conn.execute(text(f'DROP SCHEMA IF EXISTS {schema} CASCADE;'))
            else:
                # MetaData has a drop_all, but I had issues with references and needed to explicitly DROP ... CASCADE
                # So implementing here
                for table in list(self._metadata.tables.values()):
                    if only_tables and table not in only_tables: continue
                    conn.execute(text(f'DROP TABLE {table.schema}."{table.name}" CASCADE;'))
                    removed_tables.append(table)
                    self._metadata.remove(table)
                if only_tables:
                    assert list(sorted([(t.schema,t.name) for t in removed_tables]))==\
                           list(sorted([(t.schema,t.name) for t in only_tables])),\
                        f"Trouble removing {[t.name for t in only_tables]}"

    @property
    def int_schema(self):
        return CONFIG['database']['schema_names']['internal']
    @property
    def _mattab(self) -> Table:
        return self._metadata.tables[f"{self.int_schema}.Materials"]
    @property
    def _loadtab(self) -> Table:
        return self._metadata.tables[f"{self.int_schema}.Loads"]
    @property
    def _rextab(self) -> Table:
        return self._metadata.tables[f"{self.int_schema}.ReExtract"]
    @property
    def _reatab(self) -> Table:
        return self._metadata.tables[f"{self.int_schema}.ReAnalyze"]
    @property
    def _masktab(self) -> Table:
        return self._metadata.tables[f"{self.int_schema}.Masks"]
    @property
    def _diemtab(self) -> Table:
        return self._metadata.tables[f"{self.int_schema}.Dies"]
    def _mgt(self,mg,wh) -> Table:
        return self._metadata.tables.get(f"{self.int_schema}.{wh.capitalize()} -- {mg}",None)
    def _hat(self,an) -> Table:
        return self._metadata.tables.get(f"{self.int_schema}.Analysis -- {an}",None)
    @property
    def _blobtab(self) -> Table:
        return self._metadata.tables.get(f"{self.int_schema}.Blob Store")


        #@staticmethod
        #def df_to_alchemy_columns(table: pd.DataFrame):
        #    type_mapping={
        #        'int64':INTEGER,'Int64':INTEGER,
        #        'int32':INTEGER,'Int32':INTEGER,
        #        'string':VARCHAR,
        #        'float32':DOUBLE_PRECISION,'Float32':DOUBLE_PRECISION,
        #        'float64':DOUBLE_PRECISION,'Float64':DOUBLE_PRECISION,
        #        'bool':Boolean,'boolean':Boolean}
        #    return [Column(c,type_mapping[str(cdtype)],primary_key=(c=='LotWafer')) for c,cdtype in table.dtypes.items()]

class PostgreSQLDatabase(AlchemyDatabase):

    pd_to_sql_types={
        'int64':INTEGER,'Int64':INTEGER,
        'int32':INTEGER,'Int32':INTEGER,
        'string':VARCHAR,
        'str':VARCHAR,
        'float32':DOUBLE_PRECISION,'Float32':DOUBLE_PRECISION,
        'float64':DOUBLE_PRECISION,'Float64':DOUBLE_PRECISION,
        'bool':BOOLEAN,'boolean':BOOLEAN,
        'datetime64[ns]':TIMESTAMP}
    sql_to_pd_types={
        INTEGER:'Int32',
        VARCHAR:'string',
        BOOLEAN: 'bool',
        DOUBLE_PRECISION:'float64',
        TIMESTAMP:'datetime64[ns]'
    }


    def clear_excess_tables(self, conn, on_mismatch='raise',check_for_wasteful_layout_tables=False):
        layout_params=get_layout_params(conn=conn) # Don't do this until Blob Store exists!::
        # First check for "excess" old tables
        table_names=self._metadata.tables.keys()
        ins=self.int_schema+"."
        alljmptabs=set(CONFIG['measurement_groups']).union(set(CONFIG['higher_analyses']))
        excess_tables_and_comments=[]
        for table in sorted(table_names):
            if table in [f'{ins}Masks',f'{ins}Materials',f'{ins}Loads',f'{ins}ReExtract',f'{ins}ReAnalyze',f'{ins}Dies',f'{ins}Blob Store']:
                continue
            elif table.startswith(f"{ins}Meas -- ") or table.startswith(f"{ins}Sweep -- ") or table.startswith(f"{ins}Extr -- "):
                if table.split("-- ")[1] not in CONFIG['measurement_groups']:
                    excess_tables_and_comments.append((table,f"Unrecognized meas group table {table}"))
            elif table.startswith(f"{ins}Analysis -- "):
                if table.split("-- ")[1] not in CONFIG['higher_analyses']:
                    excess_tables_and_comments.append((table,f"Unrecognized analysis table {table}"))
            elif table.startswith(f"{ins}Layout -- "):
                if (g:=table.split("-- ")[1]) not in layout_params._tables_by_meas:
                    excess_tables_and_comments.append((table,f"Layout table {table} not fed by layout_params"))
                    feeding=', unfed,'
                else: feeding=''
                if check_for_wasteful_layout_tables:
                    if g in CONFIG['measurement_groups']:
                        if not CONFIG['measurement_groups'][g].get('connect_to_layout_table',True):
                            logger.warning(f"Warning: Layout table {g}{feeding} not used by its measurement group")
                    elif g in CONFIG['higher_analyses']:
                        if not CONFIG['higher_analyses'][g].get('connect_to_layout_table',False):
                            logger.warning(f"Warning: Layout table {g}{feeding} not used by its analysis")
                    else:
                        logger.warning(f"Warning: Layout table {g}{feeding} not associated with any measurement group or analysis")
            elif table.startswith("jmp."):
                if table.split(".")[1] not in alljmptabs:
                    excess_tables_and_comments.append((table,f"Unrecognized view {table}"))
            else:
                excess_tables_and_comments.append((table,f"What is table {table}?"))
        if len(excess_tables_and_comments):
            for table,comment in excess_tables_and_comments:
                if on_mismatch=='raise':
                    raise Exception(comment)
                elif on_mismatch=='warn':
                    logger.warning(comment)
                elif on_mismatch=='replace':
                    logger.warning(f"Deleting table '{table}' because: {comment}")
                    conn.execute(text(f'DROP TABLE "{table.replace(".","\".\"")}" CASCADE;'))


    def establish_database(self, on_mismatch='raise', just_metadata=False):

        class FakeConnect():
            def commit(self): pass
            def execute(self,*args,**kwargs): raise Exception("Executing on FakeConnect")
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass

        with (self.engine_connect() if not just_metadata else FakeConnect()) as conn:
            with time_it("Ensuring schema and blob store",threshold_time=.1):
                self.establish_schema_and_blob_store(conn, on_mismatch=on_mismatch, just_metadata=just_metadata)
                conn.commit()

            layout_params=get_layout_params(conn=(conn if not just_metadata else None))

            if not just_metadata:
                with time_it("Clearing excess tables", threshold_time=.1):
                    self.clear_excess_tables(conn,on_mismatch=on_mismatch)


            # Always raise if mismatch on these core tables because
            # there are foreign keys to them that will be lost if they are recreated
            with time_it("Ensuring core tables",threshold_time=.1):
                # Mask tables
                self.establish_mask_tables(conn, on_mismatch='raise',just_metadata=just_metadata)

                # Materials table
                matscheme=CONFIG['database']['materials']
                self._ensure_table_exists(conn,self.int_schema,'Materials',
                            Column('matid',INTEGER,primary_key=True,autoincrement=True),
                            *[Column(name,VARCHAR,nullable=False) for name in matscheme['info_columns'] if name!='Mask'],
                            Column(matscheme['full_name'],VARCHAR,unique=True,nullable=False),
                            Column('Mask',VARCHAR,ForeignKey("Masks.Mask",name='fk_mask',**_CASC),nullable=False),
                            Column('date_user_changed',TIMESTAMP,nullable=False),
                            on_mismatch='raise',just_metadata=just_metadata)

                # Loads table
                loadscheme=CONFIG['database']['loads']
                loadtab=self._ensure_table_exists(conn,self.int_schema,'Loads',
                          Column('loadid',INTEGER,primary_key=True,autoincrement=True),
                          Column('matid',INTEGER,ForeignKey("Materials.matid",**_CASC),nullable=False),
                          Column('MeasGroup',VARCHAR,nullable=False),
                          *[Column(name,VARCHAR,nullable=False) for name in loadscheme['info_columns']],
                          UniqueConstraint('matid','MeasGroup'),
                          on_mismatch='raise',just_metadata=just_metadata)

                # Data that's been killed from Meas/Extr
                rextab=self._ensure_table_exists(conn,self.int_schema,'ReExtract',
                          Column('matid',INTEGER,ForeignKey("Materials.matid",**_CASC),nullable=False),
                          Column('MeasGroup',VARCHAR,nullable=False),
                          Column('full_reload',BOOLEAN,nullable=False),
                          *[Column(name,VARCHAR,nullable=False) for name in loadscheme['info_columns']],
                          UniqueConstraint('matid','MeasGroup'),
                          on_mismatch='raise',just_metadata=just_metadata)

                reatab=self._ensure_table_exists(conn,self.int_schema,'ReAnalyze',
                          Column('matid',INTEGER,ForeignKey("Materials.matid",**_CASC),nullable=False),
                          Column('analysis',VARCHAR,nullable=False),
                          UniqueConstraint('matid','analysis'),
                          on_mismatch='raise',just_metadata=just_metadata)

                if not just_metadata: conn.commit()

            # TODO: when moving layout params into main config, this loop should go over CONFIG['measurement_groups']
            with time_it("Ensuring layout params tables",threshold_time=.1):
                for mgoa in layout_params._tables_by_meas:
                    if (mgoa not in CONFIG['measurement_groups']) and (mgoa not in CONFIG['higher_analyses']):
                        # TODO: reinstate this warning
                        #logger.warning(f"Ignoring request to make layout table {mgoa} because it's not connected to a meas group or analysis")
                        continue
                    self.establish_layout_parameters(layout_params,mgoa,conn,on_mismatch=on_mismatch,just_metadata=just_metadata)
                    if not just_metadata: conn.commit()
            with time_it("Ensuring measurement group tables",threshold_time=.1):
                for mg in CONFIG['measurement_groups']:
                    self.establish_measurement_group_tables(mg,conn,on_mismatch=on_mismatch,just_metadata=just_metadata)
                    if not just_metadata: conn.commit()
            with time_it("Ensuring higher_analyses tables",threshold_time=.1):
                for an in CONFIG['higher_analyses']:
                    self.establish_higher_analysis_tables(an,conn,on_mismatch=on_mismatch,just_metadata=just_metadata)
                    if not just_metadata: conn.commit()
        self._established=True

    def establish_mask_tables(self,conn, on_mismatch='raise',just_metadata=False):
        needs_update=False
        def yes_needs_update():
            nonlocal needs_update
            needs_update=True
        self._ensure_table_exists(conn,self.int_schema,f'Masks',
                          Column('Mask',VARCHAR,nullable=False,unique=True,primary_key=True),
                          Column('info_pickle',BYTEA,nullable=False),
                          on_mismatch=on_mismatch,on_init=yes_needs_update,just_metadata=just_metadata)
        self._ensure_table_exists(conn,self.int_schema,f'Dies',
                          Column('dieid',INTEGER,autoincrement=True,nullable=False,primary_key=True),
                          Column('Mask',VARCHAR,ForeignKey("Masks.Mask",**_CASC),nullable=False,index=True),
                          Column('DieXY',VARCHAR,nullable=False,index=True),
                          Column('DieRadius [mm]',INTEGER,nullable=False),
                          Column('DieCenterA [mm]',DOUBLE_PRECISION,nullable=False),
                          Column('DieCenterB [mm]',DOUBLE_PRECISION,nullable=False),
                          Column('DieX',INTEGER,nullable=False),
                          Column('DieY',INTEGER,nullable=False),
                          UniqueConstraint('Mask','DieXY'),
                          on_mismatch=on_mismatch,on_init=yes_needs_update,just_metadata=just_metadata)
        if needs_update:
            self.update_mask_info(conn)

    def fix_die_constraint(self,conn,add_or_remove='add'):
        #tab=self._mgt(mg,'Meas').name
        res=conn.execute(
            text("SELECT table_name FROM information_schema.table_constraints"
                 f" WHERE table_schema='{self.int_schema}' AND "
                 " constraint_name='fk_dieid';"))
        constrained_tabs=[x[0] for x in res]
        unconstrained_tabs=[]
        for mg,mginfo in CONFIG['measurement_groups'].items():
            if (tab:=self._mgt(mg,'Meas')) is not None:
                if tab.name not in constrained_tabs:
                    unconstrained_tabs.append(tab.name)
        for an in list(CONFIG['higher_analyses']):
            if (tab:=self._hat(an)) is not None:
                if tab.name not in constrained_tabs:
                    unconstrained_tabs.append(tab.name)
        if add_or_remove=='add':
            for tab in unconstrained_tabs:
                if 'dieid' in [c.name for c in self._metadata.tables[self.int_schema+'.'+tab].columns]:
                    conn.execute(text(f'ALTER TABLE {self.int_schema}."{tab}"' \
                                      f' ADD CONSTRAINT "fk_dieid" FOREIGN KEY ("dieid")' \
                                      f' REFERENCES {self.int_schema}."Dies" ("dieid") ON DELETE CASCADE;'))
        elif add_or_remove=='remove':
            for tab in constrained_tabs:
                if 'dieid' in [c.name for c in self._metadata.tables[self.int_schema+'.'+tab].columns]:
                    conn.execute(text(f'ALTER TABLE {self.int_schema}."{tab}"' \
                                      f' DROP CONSTRAINT "fk_dieid";'))
        else: raise ValueError(f"What is '{add_or_remove}'? Was expecting 'add' or 'remove'.")

    def update_mask_info(self,conn):
        #previous_masktab=pd.read_sql(select(*self._masktab.columns),conn)
        #######Column('Mask',VARCHAR,ForeignKey("Masks.Mask",name='fk_mask',**_CASC),nullable=False),
        diemdf=[]
        for mask,info in CONFIG['array_maps'].items():
            dbdf,to_pickle=import_modfunc(info['generator'])(**info['args'])
            diemdf.append(dbdf.assign(Mask=mask)[['Mask','DieXY','DieRadius [mm]','DieCenterA [mm]','DieCenterB [mm]','DieX','DieY']])
            update_info=dict(Mask=mask,info_pickle=pickle.dumps(to_pickle))
            conn.execute(pgsql_insert(self._masktab).values(**update_info)\
                         .on_conflict_do_update(index_elements=['Mask'],set_=update_info))

        diemdf=pd.concat(diemdf).reset_index(drop=True).reset_index(drop=False)
        previous_dietab=pd.read_sql(select(*self._diemtab.columns).order_by(self._diemtab.c['dieid']),conn).reset_index(drop=False)
        # This checks that nothing has changed in the previous table
        # very important to check that because all the measured data is only associated with a die index,
        # so if we accidentally change the die index, even by uploading the tables in a different order...
        # poof all the old data is now associated with the wrong dies or even wrong masks!!
        print("\n\n\nPREVIOUS:")
        print(previous_dietab)
        print("\n\n\nNEW:")
        print(diemdf)
        print("\n")
        assert len(previous_dietab.merge(diemdf))==len(previous_dietab),\
            "Can't add to die tables without messing up existing dies"
        self._upload_csv(diemdf.iloc[len(previous_dietab):],conn,self.int_schema,'Dies')

        self.fix_die_constraint(conn,add_or_remove='add')
        print("Successful")

    def get_mask_info(self,mask,conn=None):
        with (returner_context(conn) if conn else self.engine_connect()) as conn:
            res=conn.execute(select(self._masktab.c.info_pickle).where(self._masktab.c.Mask==mask)).all()
        assert len(res)==1, f"Couldn't get info from database about mask {mask}"
        # Must ensure restricted write access to DB since this allows arbitrary code execution
        return pickle.loads(res[0][0])

    def establish_layout_parameters(self,layout_params, measurement_group, conn, on_mismatch='raise',just_metadata=False):
        mg, df=measurement_group, layout_params._tables_by_meas[measurement_group]
        assert len(df), f"Empty layout param table for {mg}"
        assert df.index.name=='Structure'
        tabname=f"Layout -- {mg}"

        def replace_callback():
            logger.warning(f'Mismatch in layout params for "{measurement_group}", re-creating table')
            if f"{self.int_schema}.Meas -- {mg}" in self._metadata.tables:
                conn.execute(text(f'ALTER TABLE {self.int_schema}."Meas -- {mg}" '
                                  f'DROP CONSTRAINT IF EXISTS "fk_struct -- {mg}";'))
            if f"{self.int_schema}.Analysis -- {mg}" in self._metadata.tables:
                conn.execute(text(f'ALTER TABLE {self.int_schema}."Analysis -- {mg}" '
                                  f'DROP CONSTRAINT IF EXISTS "fk_struct -- {mg}";'))
            self.clear_database(only_tables=[tabname],conn=conn)
        def initialize_callback():
            self.update_layout_parameters(layout_params,mg,conn)

        with time_it(f"Ensuring layout params for {mg}",threshold_time=.005):
            cols=[Column('Structure',VARCHAR,primary_key=True),
                  *[Column(k,self.pd_to_sql_types[str(dtype)]) for k,dtype in df.dtypes.items()]]
            self._ensure_table_exists(conn,self.int_schema,f'Layout -- {mg}',*cols,
                                  on_mismatch=on_mismatch, on_init=initialize_callback, just_metadata=just_metadata)


    def _upload_binary(self, df, conn, schema, table, override_converters={}):
        with time_it(f"Conversion to binary for {table}",.1):
            bio=df_to_pgbin(df, override_converters=override_converters)
            #print("BIO len:",len(bio.read()))
            #bio.seek(0)
            #print(bio.read().hex())
            #bio.seek(0)

        with time_it(f"Upload of binary for {table}",.1):
            with conn.connection.cursor() as cur:
                cur.copy_expert(f'COPY {schema}."{table}" FROM STDIN BINARY',bio)
        ##### TEMPORARILY REMOVING DB-API COMMIT
        #conn.connection.commit()

    def _upload_csv(self, df, conn, schema, table):
        with time_it(f"Conversion to csv for {table}",.1):
            output = io.StringIO()
            df.replace([np.inf,-np.inf],np.nan).to_csv(output, sep='|', header=False, index=False)
            #print("CSV len",len(output.getvalue().encode('utf-8')))
            #output.seek(0)
            #print(output.read())
            output.seek(0)
        with time_it(f"Upload of csv for {table}",.1):
            with conn.connection.cursor() as cur:
                try:
                    cur.copy_expert(f'COPY {schema+"." if schema else ""}"{table}" FROM STDIN WITH DELIMITER \'|\' NULL \'\'',file=output)
                except Exception as e:
                    import pdb; pdb.set_trace()
                    raise
        ##### TEMPORARILY REMOVING DB-API COMMIT
        # conn.connection.commit()

    def dump_extractions(self, measurement_group, conn):
        if measurement_group not in CONFIG.measurement_groups: raise ValueError(f"Unknown group '{measurement_group}'")
        if (extr_tab:=self._mgt(measurement_group,'extr')) is None: return
        if (meas_tab:=self._mgt(measurement_group,'meas')) is None: return
        fullname=CONFIG['database']['materials']['full_name']
        conn.execute(
            pgsql_insert(self._rextab)\
                .from_select(["matid","MeasGroup",'full_reload',*ALL_LOAD_COLUMNS],
                    select(self._loadtab.c.matid,literal(measurement_group),literal(False),*[self._loadtab.c[c] for c in ALL_LOAD_COLUMNS]) \
                    .select_from(extr_tab.join(meas_tab).join(self._loadtab))\
                           .distinct())\
                .on_conflict_do_nothing())
        conn.execute(delete(extr_tab))

    def dump_measurements(self, measurement_group, conn):
        if measurement_group not in CONFIG.measurement_groups: raise ValueError(f"Unknown group '{measurement_group}'")
        try:
            meas_tab=self._mgt(measurement_group,'meas')
        except KeyError:
            return
        if meas_tab is not None:
            #raise Exception("Should this say on_conflict_do_update in case the conflict is with a previous re-extraction request?")
            conn.execute(
                pgsql_insert(self._rextab) \
                    .from_select(["matid","MeasGroup",'full_reload',*ALL_LOAD_COLUMNS],
                                 select(self._loadtab.c.matid,literal(measurement_group),literal(True),*[self._loadtab.c[c] for c in ALL_LOAD_COLUMNS]) \
                                 .select_from(meas_tab.join(self._loadtab)) \
                                 .distinct()) \
                    .on_conflict_do_nothing())
            conn.execute(delete(meas_tab))

    def dump_higher_analysis(self, analysis, conn):
        if analysis not in CONFIG.higher_analyses: raise ValueError(f"Unknown analysis '{analysis}'")
        if (an_tab:=self._hat(analysis)) is None: return
        mg=list(CONFIG.higher_analyses[analysis]['required_dependencies'])[0]
        conn.execute(
            pgsql_insert(self._reatab) \
                .from_select(["matid","analysis"],
                             select(self._loadtab.c.matid,literal(analysis)) \
                             .select_from(an_tab.join(self._loadtab,
                                                      onclause=(an_tab.c[f"loadid - {mg}"]==self._loadtab.c.loadid))) \
                             .distinct()) \
                .on_conflict_do_nothing())
        conn.execute(delete(an_tab))

    def update_layout_parameters(self, layout_params, measurement_group, conn, dump_extractions=True):
        self.establish_layout_parameters(layout_params,measurement_group,conn, on_mismatch='replace')
        tab=self._metadata.tables[f'{self.int_schema}.Layout -- {measurement_group}']
        mg=measurement_group

        conn.execute(text(f'CREATE TEMP TABLE tmplay (LIKE {self.int_schema}."Layout -- {measurement_group}");'))
        self._upload_csv(layout_params._tables_by_meas[measurement_group].reset_index(),
                         conn, None, 'tmplay')
        # https://dba.stackexchange.com/a/72642
        if conn.execute(text(
            f'''SELECT CASE WHEN EXISTS (TABLE {self.int_schema}."Layout -- {measurement_group}" EXCEPT TABLE tmplay)
              OR EXISTS (TABLE tmplay EXCEPT TABLE {self.int_schema}."Layout -- {measurement_group}")
            THEN 'different' ELSE 'same' END AS result ;''')).all()[0][0] == 'same':
            logger.debug(f"Layout parameters unchanged for {measurement_group}")
        else:
            logger.debug(f"Layout parameters changed for {measurement_group}, updating")
            if self._mgt(measurement_group,'meas') is not None:
                if dump_extractions:
                    self.dump_extractions(measurement_group,conn)
                conn.execute(text(f'ALTER TABLE {self.int_schema}."Meas -- {measurement_group}"'\
                                  f' DROP CONSTRAINT IF EXISTS "fk_struct -- {mg}";'))
            if self._hat(measurement_group) is not None:
                if dump_extractions:
                    self.dump_higher_analysis(measurement_group,conn)
                conn.execute(text(f'ALTER TABLE {self.int_schema}."Analysis -- {measurement_group}"' \
                                  f' DROP CONSTRAINT IF EXISTS "fk_struct -- {mg}";'))
            conn.execute(delete(tab))
            conn.execute(text(f'INSERT INTO {tab.schema}."{tab.name}" SELECT * from tmplay;'))
            if (self._mgt(measurement_group,'meas') is not None):
                conn.execute(text(f'ALTER TABLE {self.int_schema}."Meas -- {measurement_group}"' \
                                  f' ADD CONSTRAINT "fk_struct -- {mg}" FOREIGN KEY ("Structure")' \
                                  f' REFERENCES {self.int_schema}."{tab.name}" ("Structure") ON DELETE CASCADE;'))
            if (self._hat(measurement_group) is not None):
                conn.execute(text(f'ALTER TABLE {self.int_schema}."Analysis -- {measurement_group}"' \
                                  f' ADD CONSTRAINT "fk_struct -- {mg}" FOREIGN KEY ("Structure")' \
                                  f' REFERENCES {self.int_schema}."{tab.name}" ("Structure") ON DELETE CASCADE;'))

        conn.execute(text(f'DROP TABLE tmplay;'))
        conn.commit()


    def establish_measurement_group_tables(self,measurement_group,conn, on_mismatch='raise',just_metadata=False):
        mg, mg_info = measurement_group, CONFIG['measurement_groups'][measurement_group]

        do_recreate_view=False
        def on_init():
            nonlocal do_recreate_view
            do_recreate_view=True
        conlay=CONFIG['measurement_groups'][mg].get('connect_to_layout_table',True)
        condie=CONFIG['measurement_groups'][mg].get('connect_to_die_table',True)
        # Meas table
        def meas_replacement_callback(conn,schema,table_name,*args,**kwargs):
            # If replacing, need to drop the other tables for this meas group as well
            # since I don't want to manually recreate foreign keys
            self.dump_measurements(mg,conn)
            tabs=[self._metadata.tables.get(f'{schema}.{w} -- {mg}',None) for w in ['Meas','Extr','Sweep']]
            tabs=[tab for tab in tabs if tab is not None]
            logger.warning(f"Dumping and replacing {[tab.name for tab in tabs]}")
            self.clear_database(only_tables=tabs,conn=conn)
        self._ensure_table_exists(conn,self.int_schema,f'Meas -- {mg}',
            Column('loadid',INTEGER,ForeignKey("Loads.loadid",**_CASC),nullable=False),
            Column('measid',INTEGER,nullable=False),
            *([Column('Structure',VARCHAR,ForeignKey(f'Layout -- {mg}.Structure',
                                          name=f'fk_struct -- {mg}',**_CASC),nullable=False)] if conlay else []),
            *([Column('dieid',INTEGER,ForeignKey(f'Dies.dieid',name='fk_dieid',**_CASC),nullable=False)]
                                            if condie else []),
            Column('rawgroup',INTEGER,nullable=False),
            *[Column(k,self.pd_to_sql_types[dtype]) for k,dtype in mg_info['meas_columns'].items()],
            PrimaryKeyConstraint('loadid','measid'),
            on_mismatch=(meas_replacement_callback if on_mismatch=='replace' else on_mismatch),
            on_init=on_init, just_metadata=just_metadata)

        # Extr table
        def extr_replacement_callback(conn,schema,table_name,*args,**kwargs):
            logger.warning(f"Dumping and replacing {table_name}")
            tab=self._metadata.tables[f'{schema}.{table_name}']
            self.dump_extractions(mg,conn)
            self.clear_database(only_tables=[tab],conn=conn)
        self._ensure_table_exists(conn,self.int_schema,f'Extr -- {mg}',
            Column('loadid',INTEGER,nullable=False),
            Column('measid',INTEGER,nullable=False),
            PrimaryKeyConstraint('loadid','measid'),
            ForeignKeyConstraint(columns=['loadid','measid'],**_CASC,
                                 refcolumns=[f"Meas -- {mg}.loadid",f"Meas -- {mg}.measid",]),
            *[Column(k,self.pd_to_sql_types[dtype]) for k,dtype in mg_info['analysis_columns'].items()],
            on_mismatch=(extr_replacement_callback if on_mismatch=='replace' else on_mismatch),
            on_init=on_init, just_metadata=just_metadata)

        # Sweep table
        def sweep_replacement_callback(conn,schema,table_name,*args,**kwargs):
            logger.warning(f"Dumping and replacing {table_name}")
            tab=self._metadata.tables[f'{schema}.{table_name}']
            self.dump_measurements(mg,conn)
            self.clear_database(only_tables=[tab],conn=conn)
        self._ensure_table_exists(conn,self.int_schema,f'Sweep -- {mg}',
            Column('loadid',INTEGER,nullable=False),
            Column('measid',INTEGER,nullable=False),
            Column('sweep',BYTEA,nullable=False),
            Column('header',VARCHAR,nullable=False),
            PrimaryKeyConstraint('loadid','measid','header'),
            ForeignKeyConstraint(columns=['loadid','measid'],**_CASC,
                                 refcolumns=[f"Meas -- {mg}.loadid",f"Meas -- {mg}.measid",]),
            on_mismatch=(sweep_replacement_callback if on_mismatch=='replace' else on_mismatch), just_metadata=just_metadata)

        # TODO: Replace this with SQLAlchemy select like in get_where
        if do_recreate_view:
            layout_params=get_layout_params(conn=conn)
            logger.debug(f"(Re)-creating view for {mg}")
            view_cols=[
                CONFIG['database']['materials']['full_name'],
                *(f'Materials"."{i}' for i in CONFIG['database']['materials']['info_columns']),
                *(f'Loads"."{i}' for i in CONFIG['database']['loads']['info_columns']),
                *([f'Meas -- {mg}"."Structure'] if conlay else []),
                *([f'Dies"."DieXY',f'Dies"."DieRadius [mm]'] if condie else []),
                *mg_info['analysis_columns'].keys(),
                *mg_info['meas_columns'].keys(),
                *([c for c in layout_params._tables_by_meas[mg].columns if not c.startswith("PAD")]
                      if mg in layout_params._tables_by_meas else []),
                f'Meas -- {mg}"."loadid',
                f'Meas -- {mg}"."measid',
            ]
            view_cols=",".join([f'"{c}"' for c in view_cols])
            conn.commit()
            conn.execute(text(f'DROP VIEW IF EXISTS jmp."{mg}"; CREATE VIEW jmp."{mg}" AS SELECT {view_cols} from '\
                f'"Extr -- {mg}" '\
                f'JOIN "Meas -- {mg}" ON "Extr -- {mg}".loadid="Meas -- {mg}".loadid '\
                                   f'AND "Extr -- {mg}".measid="Meas -- {mg}".measid ' +\
                (f'JOIN "Layout -- {mg}" ON "Meas -- {mg}"."Structure"="Layout -- {mg}"."Structure" ' if conlay else '')+\
                f'JOIN "Loads" ON "Loads".loadid="Meas -- {mg}".loadid ' +\
                (f'JOIN "Dies" ON "Meas -- {mg}".dieid="Dies".dieid ' if condie else '')+\
                f'JOIN "Materials" ON "Loads".matid="Materials".matid;'))

    def establish_higher_analysis_tables(self,analysis, conn, on_mismatch='raise',just_metadata=False):
        reqlids=[Column(f'loadid - {mg}',INTEGER,ForeignKey(self._loadtab.c.loadid,**_CASC),nullable=False,index=True)
                 for mg in CONFIG.higher_analyses[analysis]['required_dependencies']]
        attlids=[Column(f'loadid - {mg}',INTEGER,ForeignKey(self._loadtab.c.loadid,**_CASC),nullable=True,index=True)
                 for mg in CONFIG.higher_analyses[analysis].get('attempt_dependencies',{})]
        condie=CONFIG['higher_analyses'][analysis].get('connect_to_die_table',True)
        conlay=CONFIG['higher_analyses'][analysis].get('connect_to_layout_table',False)
        layout_params=get_layout_params(conn=conn)
        def replacement_callback(conn,schema,table_name,*args,**kwargs):
            logger.warning(f"Dumping and replacing {table_name}")
            tab=self._metadata.tables[f'{schema}.{table_name}']
            self.dump_higher_analysis(analysis,conn)
            self.clear_database(only_tables=[tab],conn=conn)
        do_recreate_view=False
        def on_init():
            nonlocal do_recreate_view
            do_recreate_view=True
        self._ensure_table_exists(conn,self.int_schema,f'Analysis -- {analysis}',
                  *reqlids,*attlids,
                  *([Column('Structure',VARCHAR,ForeignKey(f'Layout -- {analysis}.Structure',
                       name=f'fk_struct -- {analysis}',**_CASC),nullable=False)] if conlay else []),
                  *([Column('dieid',INTEGER,ForeignKey(f'Dies.dieid',name='fk_dieid',**_CASC),nullable=False)]
                        if condie else []),
                  *[Column(k,self.pd_to_sql_types[dtype]) for k,dtype
                        in CONFIG.higher_analyses[analysis]['analysis_columns'].items()],
                  on_mismatch=(replacement_callback if on_mismatch=='replace' else on_mismatch),
                  on_init=on_init,just_metadata=just_metadata)

        # TODO: Replace this with SQLAlchemy select like in get_where
        if do_recreate_view:
            view_cols=[
                CONFIG['database']['materials']['full_name'],
                *(f'Materials"."{i}' for i in CONFIG['database']['materials']['info_columns']),
                *(f'Loads"."{i}' for i in CONFIG['database']['loads']['info_columns']),
                *([f'Analysis -- {analysis}"."Structure'] if conlay else []),
                *([f'Dies"."DieXY',f'Dies"."DieRadius [mm]'] if condie else []),
                *(CONFIG.higher_analyses[analysis]['analysis_columns'].keys()),
                *([c for c in layout_params._tables_by_meas[analysis].columns if not c.startswith("PAD")]
                  if analysis in layout_params._tables_by_meas else []),
            ]
            view_cols=",".join([f'"{c}"' for c in view_cols])
            an=analysis; mg=list(CONFIG.higher_analyses[analysis]['required_dependencies'])[0]
            conn.execute(text(f'DROP VIEW IF EXISTS jmp."{an}"; CREATE VIEW jmp."{an}" AS SELECT {view_cols} from ' \
                              f'"Analysis -- {an}" ' \
                              f'JOIN "Loads" ON "Loads".loadid="Analysis -- {an}"."loadid - {mg}"' + \
                              (f'JOIN "Layout -- {an}" ON "Analysis -- {an}"."Structure"="Layout -- {an}"."Structure" ' if conlay else '')+\
                              (f'JOIN "Dies" ON "Analysis -- {an}".dieid="Dies".dieid ' if condie else '') +\
                              f'JOIN "Materials" ON "Loads".matid="Materials".matid;'))

    def _ensure_table_exists(self, conn, schema, table_name, *args,
                             on_mismatch:Union[str,Callable]='raise',
                             on_init: Callable= (lambda : None), just_metadata=False):
        with (returner_context(conn) if (just_metadata or conn) else self.engine_connect()) as conn:
            should_be_columns=[x for x in args if isinstance(x,Column)]
            if (tab:=self._metadata.tables.get(f'{schema}.{table_name}',None)) is not None:
                try:
                    assert [c.name for c in tab.columns]==[c.name for c in should_be_columns]
                    assert [c.type.__class__ for c in tab.columns]==[c.type.__class__ for c in should_be_columns]
                except AssertionError:
                    logger.warning(f"Column mismatch in {tab.name} (note, only name and type class are checked)")
                    logger.warning(f"Currently in DB: {[(c.name,c.type.__class__.__name__) for c in tab.columns]}")
                    logger.warning(f"Should be in DB: {[(c.name,c.type.__class__.__name__) for c in should_be_columns]}")
                    if on_mismatch=='warn':
                        pass
                    elif on_mismatch=='raise':
                        raise
                    elif on_mismatch=='replace':
                        logger.warning(f"Replacing {tab.name}")
                        self.clear_database(only_tables=[tab],conn=conn)
                    elif callable(on_mismatch):
                        on_mismatch(conn,schema,table_name,*args)
                    else:
                        raise Exception(f"Can't interpret on_mismatch={on_mismatch}")
            if (tab:=self._metadata.tables.get(f'{schema}.{table_name}',None)) is None:
                if on_mismatch=='warn':
                    logger.warning(f"Need to create {table_name}")
                else:
                    tab=Table(table_name,self._metadata,*args)
                    if not just_metadata:
                        logger.info(f"Creating table {table_name}")
                        tab.create(conn)
                        on_init()
            return tab

    def dump_material(self, material_info, conn, only_meas_group=None):
        """Does not commit, so transaction will continue to have lock on Materials table."""
        fullmatname_col=CONFIG['database']['materials']['full_name']
        if only_meas_group is None:
            #statement=delete(self._mattab)\
            #                 .where(self._mattab.c[fullmatname_col]==material_info[fullmatname_col])\
            #                 .returning(self._mattab.c.date_user_changed)
            #print(conn.execute(text("EXPLAIN (ANALYZE,BUFFERS) "+str(statement.compile(compile_kwargs={'literal_binds':True})))).all())
            res=conn.execute(delete(self._mattab)\
                             .where(self._mattab.c[fullmatname_col]==material_info[fullmatname_col])\
                             .returning(self._mattab.c.date_user_changed)).all()
            if len(res): return res[0][0]
        else:
            conn.execute(delete(self._loadtab) \
                             .where(self._mattab.c[fullmatname_col]==material_info[fullmatname_col]) \
                             .where(self._mattab.c.matid==self._loadtab.c.matid)\
                             .where(self._loadtab.c.MeasGroup==only_meas_group))

    def enter_material(self, conn, user_called=True, **material_info):
        """Does not commit, so transaction will continue to have lock on Materials table."""
        fullmatname_col=CONFIG['database']['materials']['full_name']
        if not user_called:
            res=conn.execute(select(self._mattab)\
                             .where(self._mattab.c[fullmatname_col]==material_info[fullmatname_col])).all()
            if len(res):
                matid=dict(zip([c.name for c in self._mattab.columns],res[0]))['matid']
                # TODO: Could put a check here that the rest of material_info is accurate...
            else:
                raise Exception(f"While regenerating, ran across unrecognized material {material_info[fullmatname_col]}")
        else:
            update_info=material_info.copy()
            update_info.update(date_user_changed=datetime.now())
            matid=conn.execute(pgsql_insert(self._mattab)\
                               .values(**update_info)\
                               .on_conflict_do_update(index_elements=[fullmatname_col],set_=update_info)\
                               .returning(self._mattab.c.matid))\
                        .all()[0][0]
        return matid

    def push_data(self, matload_info, data_by_meas_group:dict,
                  clear_all_from_material=True, user_called=True, re_extraction=False,
                  defer_analyses=False):
        """
        Notes
        -----
        `re_extraction=True` should only be used by internal code healing the database
        (ie after a table has been dropped). If `re_extraction=True`, `push_data` will
        make no effort to clear out prior data, so abuse of this can result in uniqueness
        violation errors.
        """
        fullmatname_col=CONFIG['database']['materials']['full_name']
        assert not (user_called and re_extraction), "Re-extraction is not a user-update"
        material_info={k:v for k,v in matload_info.items() if k in ALL_MATERIAL_COLUMNS}
        load_info={k:v for k,v in matload_info.items() if k in ALL_LOAD_COLUMNS}

        with self.engine_connect() as conn:
            conn.execute(text(f"SET SEARCH_PATH={self.int_schema};"))

            # If clear material, then drop this material from the database before beginning
            date_user_changed=None
            if clear_all_from_material:
                assert not re_extraction, "Doesn't make sense to clear material when re-extracting"
                with time_it("Dropping all from material"):
                    date_user_changed=self.dump_material(material_info, conn)

            # Now ensure this material is in the database
            matid=self.enter_material(conn,**material_info, user_called=user_called,
                                      date_user_changed=date_user_changed)

            # Invalidate the relevant analyses
            insrt=[]
            analyses=CONFIG.get_dependent_analyses(list(data_by_meas_group.keys()))
            if len(analyses):
                for an in analyses:
                    insrt.append(
                        str(pgsql_insert(self._reatab) \
                            .values(matid=matid,analysis=an) \
                            .on_conflict_do_nothing()\
                            .compile(compile_kwargs={'literal_binds':True})))
                conn.execute(text("; ".join(insrt)))


            # For each meas_group
            collected_loadids={}
            for meas_group, mt_or_df in data_by_meas_group.items():

                if not re_extraction:
                    # Drop any previous loads from the Loads table
                    # (if clear_material, this is already handled by dump_material above)
                    if not clear_all_from_material:
                        self.dump_material(material_info, conn, only_meas_group=meas_group)

                    # Put an entry into the Loads table and get the loadid
                    loadid=conn.execute(pgsql_insert(self._loadtab)\
                                       .values(matid=matid,MeasGroup=meas_group,**load_info)\
                                       .returning(self._loadtab.c.loadid))\
                                .all()[0][0]

                #print(type(mt_or_df),isinstance(mt_or_df, MeasurementTable))
                from datavac.io.measurement_table import MeasurementTable
                if isinstance(mt_or_df, MeasurementTable):
                    analysis_cols=list(CONFIG['measurement_groups'][meas_group]['analysis_columns'])
                    meas_cols=list(CONFIG['measurement_groups'][meas_group]['meas_columns'])

                    df=mt_or_df._dataframe

                    if not re_extraction:
                        # Upload the measurement list
                        with time_it(f"Meas table {meas_group} altogether"):
                            mask=material_info['Mask']
                            condie=CONFIG['measurement_groups'][meas_group].get('connect_to_die_table',True)
                            conlay=CONFIG['measurement_groups'][meas_group].get('connect_to_layout_table',True)
                            if condie:
                                diem=pd.DataFrame.from_records(conn.execute(select(self._diemtab.c.DieXY,self._diemtab.c.dieid)\
                                                .where(self._diemtab.c.Mask==mask)).all(),columns=['DieXY','dieid'])
                                df2=df.reset_index() \
                                    .assign(loadid=loadid,Mask=mask).rename(columns={'index':'measid'}) \
                                    [['loadid','measid',*(['Structure'] if conlay else []),'DieXY','rawgroup',*meas_cols]].merge(diem,how='left',on='DieXY')\
                                    [['loadid','measid',*(['Structure'] if conlay else []),'dieid','rawgroup',*meas_cols]]
                            else:
                                df2=df.reset_index() \
                                    .assign(loadid=loadid,Mask=mask).rename(columns={'index':'measid'})\
                                    [['loadid','measid',*(['Structure'] if conlay else []),'rawgroup',*meas_cols]]
                            self._upload_csv(df2,conn,self.int_schema,self._mgt(meas_group,'meas').name)

                        # Upload the raw sweep
                        meas_type=CONFIG.get_meas_type(meas_group)
                        sweepconv=(pd_to_pg_converters['STRING']) \
                            if (hasattr(meas_type,'ONESTRING') and meas_type.ONESTRING) else lambda s: s.tobytes()
                        self._upload_binary(
                            df[mt_or_df.headers]
                                .stack().reset_index()
                                .assign(loadid=loadid).rename(columns={'level_0':'measid','level_1':'header',0:'sweep'}) \
                                [['loadid','measid','sweep','header']],
                            conn,self.int_schema,self._mgt(meas_group,'sweep').name,
                            override_converters={'sweep':sweepconv,'header':pd_to_pg_converters['STRING']}
                        )

                    # Upload the extracted values
                    if not re_extraction:
                        df=df.assign(loadid=loadid).reset_index().rename(columns={'index':'measid'})
                    assert len(ulid:=df['loadid'].unique())==1
                    try:
                        df=df[['loadid','measid',*analysis_cols]]
                    except KeyError as e:
                        logger.warning(f"Missing columns: {str(e)}")
                        logger.warning(f"Present columns are {list(df.columns)}")
                        raise e
                    loadid=int(ulid[0])
                    try:
                        self._upload_csv(
                            df,
                            conn,self.int_schema,self._mgt(meas_group,'extr').name
                        )
                    except Exception as e:
                        print("OOPS")
                        raise e

                    # If we've succeeded thus far, we can drop this matid, MeasGroup from the refreshes table
                    dstat=delete(self._rextab)\
                        .where(self._rextab.c.MeasGroup==meas_group)\
                        .where(self._rextab.c.matid==self._loadtab.c.matid) \
                        .where(self._loadtab.c.loadid==loadid)
                    if re_extraction:
                        dstat=dstat.where(self._rextab.c.full_reload==False)
                    #print(dstat.compile())
                    conn.execute(dstat)

                    collected_loadids[meas_group]=loadid

            if not defer_analyses:
                self.perform_analyses(conn, analyses,
                                  precollected_data_by_meas_group=data_by_meas_group,
                                  precollected_loadids=collected_loadids,
                                  precollected_matid=matid)
            conn.commit()
            logger.debug(f"Completed all tasks for {str(matload_info)}")

    def perform_analyses(self, conn, analyses,
                         precollected_data_by_meas_group={}, precollected_loadids={}, precollected_matid=None):
        mg_to_data=precollected_data_by_meas_group
        matid=precollected_matid
        mask=conn.execute(select(self._mattab.c.Mask).where(self._mattab.c.matid==matid)).all()[0][0]
        diem=pd.DataFrame.from_records(conn.execute(
            select(self._diemtab.c.DieXY,self._diemtab.c.dieid) \
                .where(self._diemtab.c.Mask==mask)).all(),columns=['DieXY','dieid'])
        for an in analyses:
            condie=CONFIG['higher_analyses'][an].get('connect_to_die_table',True)
            conlay=CONFIG['higher_analyses'][an].get('connect_to_layout_table',False)
            if all(k in mg_to_data for k in CONFIG.higher_analyses[an]['required_dependencies']):
                logger.debug(f"Running analysis: {an}")
            else:
                logger.debug(f"Skipping analysis {an} because missing some required dependencies")
                continue
            df=import_modfunc(CONFIG.higher_analyses[an]['analysis_func'])(
                #**{v: mg_to_data[k].scalar_table_with_layout_params() for k,v in
                **{v: mg_to_data[k] for k,v in
                   CONFIG.higher_analyses[an]['required_dependencies'].items()},
                #**{v: mg_to_data.get(k,None).scalar_table_with_layout_params() for k,v in
                **{v: mg_to_data.get(k,None) for k,v in
                   CONFIG.higher_analyses[an].get('attempt_dependencies',{}).items()})
            if df is None:
                logger.debug(f"No data for analysis {an}")
                continue

            loadids=dict(
                **{f'loadid - {mg}':precollected_loadids[mg] for mg in
                   CONFIG.higher_analyses[an]['required_dependencies']}, \
                **{f'loadid - {mg}':precollected_loadids.get(mg,None) for mg in
                   CONFIG.higher_analyses[an].get('attempt_dependencies',{})})


            #if not re_extraction:
            #    # Upload the measurement list
            #    with time_it("Meas table altogether"):
            #        mask=material_info['Mask']
            #        diem=pd.DataFrame.from_records(conn.execute(select(self._diemtab.c.DieXY,self._diemtab.c.dieid) \
            #                                                    .where(self._diemtab.c.Mask==mask)).all(),columns=['DieXY','dieid'])
            #        df2=df.reset_index() \
            #            .assign(loadid=loadid,Mask=mask).rename(columns={'index':'measid'}) \
            #            [['loadid','measid','Structure','DieXY','rawgroup',*meas_cols]].merge(diem,how='left',on='DieXY') \
            #            [['loadid','measid','Structure','dieid','rawgroup',*meas_cols]]
            #        self._upload_csv(df2,conn,self.int_schema,self._mgt(meas_group,'meas').name)

            if condie:
                df=df.merge(diem,how='left',on='DieXY')
            df=df.assign(**loadids)
            self._upload_csv(
                df[[*(loadids.keys()),*(['Structure'] if conlay else []),*(['dieid'] if condie else []),*CONFIG.higher_analyses[an]['analysis_columns']]],
                conn,self.int_schema,self._hat(an).name
            )

            conn.execute(delete(self._reatab)\
                .where(self._reatab.c.matid==matid)\
                .where(self._reatab.c.analysis==an))

            ## Invalidate the relevant analyses
            #insrt=""
            #for an in CONFIG.get_dependent_analyses(list(data_by_meas_group.keys())):
            #    insrt+= \
            #        str(pgsql_insert(self._reatab) \
            #            .values(matid=matid,analysis=an) \
            #            .compile(compile_kwargs={'literal_binds':True}))
            #conn.execute(text(insrt))



    def get_data_for_regen(self, meas_group, matname, on_no_data='raise', conn=None):
        meas_cols=list(CONFIG['measurement_groups'][meas_group]['meas_columns'])
        die_cols=(['DieXY','DieCenterA [mm]','DieCenterB [mm]'] if CONFIG['measurement_groups'][meas_group].get('connect_to_die_table',True) else [])
        laycols=(['Structure'] if CONFIG['measurement_groups'][meas_group].get('connect_to_layout_table',True) else [])
        data=self.get_data(meas_group=meas_group,
                      # TODO: BAD hardcoding of columns
                      scalar_columns=['loadid','measid','rawgroup',*ALL_MATLOAD_COLUMNS,*die_cols,*laycols,*meas_cols],
                      include_sweeps=True, raw_only=True, unstack_headers=True,conn=conn,
                      **{CONFIG['database']['materials']['full_name']:[matname]})

        if not(len(data)):
            match on_no_data:
                case 'raise':
                    raise Exception(f"No data for re-extraction of {matname} with measurement group {meas_group}")
                case None:
                    return None


        headers=[]
        for c in data.columns:
            if c=='rawgroup': break
            headers.append(c)

        from datavac.io.measurement_table import MultiUniformMeasurementTable, UniformMeasurementTable

        meas_type=CONFIG.get_meas_type(meas_group)

        # This uses the fact that there is only one loadid for a given mg and matname
        return MultiUniformMeasurementTable([
            UniformMeasurementTable(dataframe=df.reset_index(drop=True),headers=headers,
                                    meas_length=None,meas_type=meas_type,meas_group=meas_group)
            for rg, df in data.groupby('rawgroup')])

    def get_data(self,meas_group,scalar_columns=None,include_sweeps=False,
                         unstack_headers=False,raw_only=False,conn=None,**factors):
        if meas_group in CONFIG.higher_analyses:
            assert (include_sweeps==False or include_sweeps==[]), f"For analysis (eg {meas_group}), include_sweeps must be False or [], not {include_sweeps}"
            #assert unstack_headers==False
            assert raw_only==False
            return self.get_data_from_analysis(meas_group,scalar_columns=scalar_columns,conn=conn,**factors)
        elif meas_group in CONFIG.measurement_groups:
            return self.get_data_from_meas_group(meas_group,scalar_columns=scalar_columns,include_sweeps=include_sweeps,
                 unstack_headers=unstack_headers,raw_only=raw_only,conn=conn,**factors)
        else:
            raise Exception(f"What is '{meas_group}'?")


    def get_data_from_analysis(self,analysis,scalar_columns=None, conn=None, **factors):
        conlay=CONFIG['higher_analyses'][analysis].get('connect_to_layout_table',False)
        condie=CONFIG['higher_analyses'][analysis].get('connect_to_die_table',True)
        anlytab=self._hat(analysis)
        if conlay: layotab=self._mgt(analysis,'layout')
        involved_tables=([anlytab]+ \
                         [*([self._diemtab] if condie else []),self._loadtab,self._mattab]+ \
                         ([layotab] if conlay else []))
        all_cols=[c for tab in involved_tables for c in tab.columns]
        def get_col(cname):
            try: return next(c for c in all_cols if c.name==cname)
            except StopIteration:
                raise Exception(f"Couldn't find column {cname} among {[c.name for c in all_cols]}")
        if scalar_columns:
            selcols=[get_col(sc) for sc in scalar_columns]
        else:
            selcols=list(set([get_col(sc.name) for sc in all_cols]))

        mg=list(CONFIG.higher_analyses[analysis]['required_dependencies'])[0]
        thejoin=functools.reduce((lambda x,y: x.join(y)
                                    if y is not self._loadtab
                                    else x.join(y,onclause=(anlytab.c[f"loadid - {mg}"]==self._loadtab.c.loadid))),
                                 involved_tables)
        sel=select(*selcols).select_from(thejoin)
        sel=functools.reduce((lambda s, f: s.where(get_col(f).in_(factors[f]))), factors, sel)
        with (returner_context(conn) if conn else self.engine_connect()) as conn:
            data=pd.read_sql(sel,conn,dtype={c.name:self.sql_to_pd_types[c.type.__class__] for c in selcols
                                             if c.type.__class__ in self.sql_to_pd_types})
        return data

    def get_data_from_meas_group(self,meas_group,scalar_columns=None,include_sweeps=False,
                 unstack_headers=False,raw_only=False,conn=None,**factors):
        meastab=self._mgt(meas_group,'meas')
        sweptab=self._mgt(meas_group,'sweep')
        extrtab=self._mgt(meas_group,'extr')
        layotab=self._mgt(meas_group,'layout')
        diemtab=self._diemtab if CONFIG['measurement_groups'][meas_group].get('connect_to_die_table',True) else None

        if include_sweeps not in [True,False]:
            if len(include_sweeps):
                factors['header']=include_sweeps
            else:
                include_sweeps=False
        involved_tables=(([extrtab] if not raw_only else [])+\
                            [meastab,diemtab,layotab,self._loadtab,self._mattab]\
                        +([sweptab] if include_sweeps else []))
        involved_tables=[t for t in involved_tables if t is not None]
        all_cols=[c for tab in involved_tables for c in tab.columns]
        def get_col(cname):
            try:
                return next(c for c in all_cols if c.name==cname)
            except StopIteration:
                raise Exception(f"Couldn't find column {cname} among {[c.name for c in all_cols]} from {[i.name for i in involved_tables]}")
        if scalar_columns:
            if (include_sweeps and unstack_headers):
                for sc in ['loadid','measid']:
                    if sc not in scalar_columns: scalar_columns=scalar_columns+[sc] # not inplace
            selcols=[get_col(sc) for sc in scalar_columns]
        else:
            selcols=list(set([get_col(sc.name) for sc in all_cols if sc.table is not sweptab]))
        selcols+=([sweptab.c.sweep,sweptab.c.header] if include_sweeps else [])

        ## TODO: Be more selective in thejoin
        thejoin=functools.reduce((lambda x,y: x.join(y)),involved_tables)
        sel=select(*selcols).select_from(thejoin)
        sel=functools.reduce((lambda s, f: s.where(get_col(f).in_(factors[f]))), factors, sel)

        return self._get_data_from_meas_group_helper(meas_group,sel=sel,selcols=selcols,include_sweeps=include_sweeps,
                                              unstack_headers=unstack_headers,raw_only=raw_only,conn=conn)

    def _get_data_from_meas_group_helper(self,meas_group,sel,selcols,include_sweeps=False,
                                         unstack_headers=False,raw_only=False,conn=None) -> pd.DataFrame:
        with (returner_context(conn) if conn else self.engine_connect()) as conn:
            with time_it("Actual read_sql",threshold_time=.03):
                data=pd.read_sql(sel,conn,dtype={c.name:self.sql_to_pd_types[c.type.__class__] for c in selcols
                                                if c.type.__class__ in self.sql_to_pd_types})
        if 'sweep' in data:
            with time_it("Sweep decoding",threshold_time=.03):
                meas_type=CONFIG.get_meas_type(meas_group)
                if not (hasattr(meas_type,'ONESTRING') and meas_type.ONESTRING):
                    for h in list(data['header'].unique()):
                        assert meas_type.get_preferred_dtype(h) in [np.float32,'onestring'],\
                            "Haven't dealt with sweeps that aren't float32 or 'onestring'"
                    data['sweep']=data['sweep'].map(functools.partial(np.frombuffer, dtype=np.float32))
                else:
                    data['sweep']=data['sweep'].map(lambda x: x.decode('utf-8'))
            if include_sweeps and unstack_headers:
                with time_it("Sweep unstacking",threshold_time=.03):
                    unstacking_indices= ['loadid','measid'] if unstack_headers is True else unstack_headers
                    data=self._unstack_header_helper(data,unstacking_indices, drop_index=(not raw_only))
        return data

    def get_meas_data_for_jmp(self,meas_group,loadids,measids):
        # https://stackoverflow.com/a/6672707
        values="(VALUES "+",".join([f"({l},{m})" for l,m in zip(loadids,measids)])+") AS t2 (loadid, measid)"
        query=f"SELECT t1.loadid, t1.measid, sweep, header from {self.int_schema}.\"Sweep -- {meas_group}\""\
              f" AS t1 JOIN {values} ON t1.loadid = t2.loadid AND t1.measid = t2.measid"

        data= self._get_data_from_meas_group_helper(meas_group,sel=query,selcols={},include_sweeps=True,
                                              unstack_headers=True, raw_only=True)

        from datavac.util.tables import stack_multi_sweeps
        from datavac.util.util import only
        headers=[k for k in data.columns if k not in ['loadid','measid']]
        if all(('@' not in k) for k in headers):
            # Assume column names of form X, Y... doesn't actually matter which is independent, so say first
            x=headers[0]
            swvs=[]
            ys_withdir=headers[1:]
            #data=data.rename(columns=dict(zip(ys,[y+"@" for y in headers[1:]])))
        else:
            # Assume column names of form X, fY1@SWVR=val, fY2@SWVR=val ...
            possible_xs=[k for k in headers if '@' not in k]
            x=only(possible_xs,f"None or multiple possible x values in {possible_xs}")
            swvs=list(set([k.split("@")[1].split("=")[0] for k in headers if '@' in k]))
            ys_withdir=[k.split("@")[0] for k in headers if '@' in k]
        directed=all(y[0] in ('f','r') for y in ys_withdir)
        ys=list(set([y[1:] for y in ys_withdir] if directed else ys_withdir))
        ys=[y for y in ys if y not in swvs]
        #print(x,ys,swvs)
        #print(data)
        data=stack_multi_sweeps(data,x=x,ys=ys,swvs=swvs,restrict_dirs=(('f','r') if directed else ('',)))
        print(x,ys)
        data = data.explode([x,*ys],ignore_index=True)\
            [['loadid','measid',*[k for k in data.columns if k not in ['loadid','measid']]]]
        #print(data)
        #print(data.columns)
        return data


    @staticmethod
    def _unstack_header_helper(data,unstacking_indices, drop_index=True):
        sweep_part=data[[*unstacking_indices,'header','sweep']] \
            .pivot(index=unstacking_indices,columns='header',values='sweep')
        other_part=data.drop(columns=['header','sweep']) \
            .drop_duplicates(subset=unstacking_indices).set_index(unstacking_indices)
        return pd.merge(sweep_part,other_part,how='inner',
                        left_index=True,right_index=True,validate='1:1').reset_index(drop=drop_index)

    @staticmethod
    def joined_table(factor_names:list[str],absolute_needs:list[Table],
                     table_depends:dict[Table,list[Table]], pre_filters:dict[str,list]):
        """ Creates an SQL Alchemy Select joining the tables needed to get the desired factors

        Args:
            factor_names: list of column names to be selected
            absolute_needs: list of tables which must be included in the join
            table_depends: mapping of dependencies which tables have on other tables to be joined
            pre_filters: filters (column name to list of allowed values) to apply to the data before joining

        Returns:
            An SQLAlchemy Select
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
        factor_cols:list[Column]=[get_col(f) for f in factor_names]
        def apply_joins():
            ordered_needed_tables=[]
            need_queue=set(absolute_needs+[f.table for f in factor_cols])
            while len(need_queue):
                for n in need_queue.copy():
                    further_needs=table_depends[n]
                    if n in ordered_needed_tables:
                        need_queue.remove(n)
                    elif all(pn in ordered_needed_tables for pn in further_needs):
                        ordered_needed_tables.append(n)
                        need_queue.remove(n)
                    else: need_queue|=set(further_needs)
            return functools.reduce((lambda x,y: x.join(y)),ordered_needed_tables)
        return apply_wheres(select(*factor_cols).select_from(apply_joins()))

    def get_factors(self,meas_group_or_analysis,factor_names,pre_filters={}):

        # Assess inputs
        if (mgoa:=meas_group_or_analysis) in CONFIG['measurement_groups']: which='measurement_groups'
        elif mgoa in CONFIG['higher_analyses']: which='higher_analyses'
        else: raise Exception(f"Couldn't identify '{meas_group_or_analysis}' as a meas_group or analysis")

        condie=CONFIG[which][mgoa].get('connect_to_die_table',True)
        conlay=CONFIG[which][mgoa].get('connect_to_layout_table',(False if which=='higher_analyses' else True))

        # Tables we may need
        table_depends={}
        if which == 'measurement_groups':
            table_depends[coretab:=(meastab:=self._mgt(mgoa,'meas'))]=[]
            table_depends[extrtab:=self._mgt(mgoa,'extr')]=[meastab]
            #sweptab=self._mgt(mgoa,'sweep'); table_depends
        if which == 'higher_analyses':
            table_depends[coretab:=(anlstab:=self._hat(mgoa))]=[]
        table_depends[self._loadtab]=[]
        table_depends[self._mattab]=[self._loadtab]
        if condie: table_depends[self._diemtab]=[self._mattab]
        if conlay: table_depends[layotab:=self._mgt(mgoa,'layout')]=[coretab]
        assert not any(t is None for t in table_depends)

        # Define a select with the pre_filtered data and set it up as a CTE named tmp
        bigtable=self.joined_table(factor_names=factor_names,absolute_needs=[coretab],
                                   table_depends=table_depends,pre_filters=pre_filters)\
            .compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        query="WITH tmp AS ("+str(bigtable)+")\n"

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
            query+=f"""(SELECT ROW_NUMBER() over (order by  null) as "ind___{f}", tmp."{f}" """\
                                                            f""" FROM tmp GROUP BY tmp."{f}") t{i}\n"""
            if i!=0: query+=f""" ON t0."ind___{factor_names[0]}"=t{i}."ind___{f}" """
            if i!=len(factor_names)-1: query+=" FULL OUTER JOIN "

        with self.engine_connect() as conn:
            with time_it("Executing SQL in get_factors",threshold_time=.01):
                records=pd.read_sql(query,conn)
        if not len(records): return {f:[] for f in factor_names}
        else: return {f:list(records.loc[records[f'ind___{f}'].notna(),f]) for f in factor_names}


    def establish_schema_and_blob_store(self, conn=None, on_mismatch='raise', just_metadata=False):
        if not just_metadata:
            with (returner_context(conn) if conn else self.engine_connect()) as conn1:
                make_schemas=" ".join([f"CREATE SCHEMA IF NOT EXISTS {schema}; " \
                                       f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO PUBLIC; " \
                                       f"GRANT USAGE ON SCHEMA {schema} TO PUBLIC; " \
                                       f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT ON TABLES TO PUBLIC; "
                                       for schema in CONFIG['database']['schema_names'].values()])
                #conn1.execute(text(make_schemas))
                conn1.execute(text(make_schemas+f"SET SEARCH_PATH={self.int_schema};"))
                #conn1.execute(text(f"ALTER DATABASE {get_db_connection_info()['Database']} SET search_path TO {self.int_schema};"))
                ## If we just made the schema for the first time, we need to re-reflect
                #if self.int_schema not in self._metadata._schemas:
                #    conn1.commit()
                #    self._metadata.reflect(conn1)


        self._ensure_table_exists(conn,self.int_schema,'Blob Store',
                                  Column('name',VARCHAR,primary_key=True),
                                  Column('blob',BYTEA,nullable=False),
                                  Column('date_stored',TIMESTAMP,nullable=False),
                                  on_mismatch=on_mismatch, just_metadata=just_metadata)

    def store_obj(self,name,obj,conn=None):
        with (returner_context(conn) if conn else self.engine_begin()) as conn:
            update_info=dict(name=name,blob=pickle.dumps(obj),date_stored=datetime.now())
            conn.execute(pgsql_insert(self._blobtab).values(**update_info) \
                         .on_conflict_do_update(index_elements=['name'],set_=update_info))
    def get_obj(self,name, conn=None):
        with time_it(f"Getting {name} from DB",threshold_time=.1):
            with (returner_context(conn) if conn else self.engine_begin()) as conn:
                res=list(conn.execute(select(self._blobtab.c.blob).where(self._blobtab.c.name==name)).all())
        if not len(res): raise KeyError(name)
        assert len(res)==1
        with time_it(f"Unpickling {name} from DB",threshold_time=.1):
            print(len(res[0][0]))
            return pickle.loads(res[0][0])
    def get_obj_date(self,name, conn=None):
        with time_it(f"Getting '{name}' date from DB",threshold_time=.001):
            with (returner_context(conn) if conn else self.engine_begin()) as conn:
                res=list(conn.execute(select(self._blobtab.c.date_stored).where(self._blobtab.c.name==name)).all())
        if not len(res): raise KeyError(name)
        assert len(res)==1
        return res[0][0]

    def run_query(self,query,conn=None, commit=False):
        with (returner_context(conn) if conn else self.engine_connect()) as conn:
            result=conn.execute(text(query)).all()
            if commit: conn.commit()
        return result
    def read_sql(self,query,conn=None):
        with (returner_context(conn) if conn else self.engine_connect()) as conn:
            result=pd.read_sql(query,conn)
        return result

##### NOT FUNCTIONAL
####class SQLiteDatabase(AlchemyDatabase):
####
####    def get_engine(self):
####        folder:Path=Path(os.environ['DATAVACUUM_CACHE_DIR'])/"db"
####        assert "//" not in str(folder) and "\\\\" not in str(folder), \
####            f"DATAVACUUM_CACHE_DIR points to a remote directory [{os.environ['DATAVACUUM_CACHE_DIR']}].  " \
####            "This would be miserably slow to use for SQLITE."
####        folder.mkdir(exist_ok=True)
####        self._sync_engine=create_engine(f"sqlite:///{folder}/SummaryData.db")

def cli_clear_database(*args):
    parser=argparse.ArgumentParser(description='Clears out the database [after confirming].')
    parser.add_argument('-y','--yes',action='store_true',help="Don't ask confirmation, just clear it.")
    parser.add_argument('-t','--table',action='append',help="Clear specific table(s), eg -t TAB1 -t TAB2")
    parser.add_argument('-s','--schema_also',action='store_true',help="Drop schemas as well")
    parser.add_argument('--keep_rex',action='store_true',help="Keep the materials and re-ex tables (ie leave enough info to heal)")
    namespace=parser.parse_args(args)

    db=get_database(metadata_source='reflect')
    try:
        only_tables=list(db._metadata.tables.values()) if namespace.table is None else [db._metadata.tables[t] for t in namespace.table]
    except KeyError as e:
        logger.critical(f"Couldn't find {str(e)}.")
        if '.' not in str(e):
           logger.critical("Did you forget to include the schema?")
        logger.critical(f"Options in include {list(db._metadata.tables.keys())}")
        return
    if namespace.keep_rex:
        only_tables=[t for t in only_tables
                     if t.name not in ["Materials",f"ReExtract"]]
    if namespace.schema_also:
        logger.info(f"Will drop all DataVacuum schemas.")
    else:
        logger.info(f"Tables that will get cleared: {sorted([t.name for t in only_tables])}")
    if namespace.yes \
            or (input('Are you sure you want to clear the database? ').strip().lower()=='y'):
        logger.warning("Clearing database")
        db.clear_database(only_tables=only_tables,schema_also=namespace.schema_also)
        logger.warning("Done clearing database")

#def cli_update_diemaps(*args):
#    parser=argparse.ArgumentParser()
#    namespace=parser.parse_args(args)

def cli_update_layout_params(*args):
    parser=argparse.ArgumentParser(description='Updates layout params in database.')
    parser.add_argument('-sr','--skip-regenerate',action='store_true',help='Skip regenerating layout parameters')
    namespace=parser.parse_args(args)

    db=get_database(metadata_source='reflect')
    layout_params=get_layout_params(force_regenerate=(not namespace.skip_regenerate))
    with db.engine_connect() as conn:
        for mg in layout_params._tables_by_meas:
            db.update_layout_parameters(layout_params,mg,conn)
        conn.commit()

def cli_dump_measurement(*args):
    parser=argparse.ArgumentParser(description='Dumps measurements')
    parser.add_argument('-g','--group',action='append',help='Measurement group(s) to drop, eg -g GROUP1 -g GROUP2')
    namespace=parser.parse_args(args)

    db=get_database(metadata_source='reflect')
    with db.engine_connect() as conn:
        for mg in (namespace.group if namespace.group else CONFIG.measurement_groups):
            db.dump_measurements(mg,conn)
        conn.commit()

def cli_dump_extraction(*args):
    parser=argparse.ArgumentParser(description='Dumps extractions')
    parser.add_argument('-g','--group',action='append',help='Measurement group(s) to drop, eg -g GROUP1 -g GROUP2')
    namespace=parser.parse_args(args)

    db=get_database(metadata_source='reflect')
    with db.engine_connect() as conn:
        for mg in (namespace.group if namespace.group else CONFIG.measurement_groups):
            db.dump_extractions(mg,conn)
        conn.commit()

def cli_dump_analysis(*args):
    parser=argparse.ArgumentParser(description='Dumps analysis')
    parser.add_argument('-a','--analysis',action='append',help='Analysis to drop, eg -g ANALYSIS1 -g ANALYSIS2')
    namespace=parser.parse_args(args)

    db=get_database(metadata_source='reflect')
    with db.engine_connect() as conn:
        for an in (namespace.analysis if namespace.analysis else CONFIG.higher_analyses):
            print(f"Dumping {an}")
            db.dump_higher_analysis(an,conn)
        conn.commit()

def cli_force_database(*args):
    parser=argparse.ArgumentParser(description='Replaces any tables not currently in agreement with schema')
    parser.add_argument('-dr','--dry_run',action='store_true',help='Just print what would be done')
    namespace=parser.parse_args(args)
    db=get_database(metadata_source='reflect')
    db.validate(on_mismatch=('warn' if namespace.dry_run else 'replace'))

def cli_upload_data(*args):
    from datavac.io.meta_reader import ALL_MATLOAD_COLUMNS

    parser=argparse.ArgumentParser(description='Extracts and uploads data')
    for col in ALL_MATLOAD_COLUMNS:
        parser.add_argument(f'--{col.lower()}',action='append',
            help=f"Restrict to specified {col}(s): "\
                 f"eg --{col.lower()} {col.upper()}1 --{col.lower()} {col.upper()}2")
    parser.add_argument('-f','--folder',action='append',
                        help=f"Restrict to specified folder(s): " \
                             f"eg -f FOLDER1 -f FOLDER2.  Default is all of {os.environ['DATAVACUUM_READ_DIR']}")
    parser.add_argument('-g','--group',action='append',help='Restrict to measurement group(s), eg -g GROUP1 -g GROUP2')
    parser.add_argument('-k','--keep_other_groups',action='store_true',
                        help=f"Keep measurement data related measurement groups that are not present in this upload.  "
                             f"(Default is to drop all data related to a given material when uploading afresh).")
    parser.add_argument('-y','--yes',action='store_true',help='Don\'t ask for confirmation, just do it.')
    parser.add_argument('-i','--isolate_errors',action='store_true',
                        help='Split by top-level folder if one folder has an error, others will continue.')

    namespace=parser.parse_args(args)

    folders=namespace.folder
    only_material={col: getattr(namespace,col.lower()) for col in ALL_MATLOAD_COLUMNS}
    only_material={k:v for k,v in only_material.items() if v is not None}

    db=get_database()
    read_and_upload_data(db, folders, only_material=only_material, only_meas_groups=namespace.group,
        clear_all_from_material=(not namespace.keep_other_groups),user_called=True,
                         yes=namespace.yes, isolate_errors=namespace.isolate_errors)

def read_and_upload_data(db,folders=None,only_material={},only_meas_groups=None,
                         clear_all_from_material=True, user_called=True, cached_glob=None,
                         yes=False, isolate_errors=False):
    from datavac.io.meta_reader import read_and_analyze_folders

    if folders is None:
        if ((connected:=CONFIG.meta_reader.get('connect_toplevel_folder_to',None)) and (connected in only_material)):
            folders=only_material[connected]
        else:
            if not yes:
                if not (input(f'No folder or {connected} restriction, continue to read EVERYTHING? [y/n] ').strip().lower()=='y'):
                    return
            folders=[f.name for f in Path(os.environ['DATAVACUUM_READ_DIR']).glob('*') if 'IGNORE' not in f.name]

    if isolate_errors:
        timings={}
        failures={}
        all_start_time=time.time()
        for folder in folders:
            start_time=time.time()
            try:
                read_and_upload_data(db,[folder],only_material=only_material,only_meas_groups=only_meas_groups,
                                     clear_all_from_material=clear_all_from_material,user_called=user_called,
                                     cached_glob=cached_glob,yes=yes,isolate_errors=False)
            except Exception as e:
                logger.critical(f"Failed on {folder}")
                logger.critical(traceback.format_exc())
                failures[folder]=str(e)
            end_time=time.time()
            timings[folder]=end_time-start_time
        all_end_time=time.time()
        print("##############################################")
        print(f"Complete in {(all_end_time-all_start_time)/60/60:.2f} hours")
        print("##### Timings")
        for f,t in timings.items():
            print(f"{f}: {t:.1f} seconds")
        print("##### Failures")
        for f,fail in failures.items():
            print(f"{f}: {fail}")
        with open("upload_all.pkl",'wb') as f:
            pickle.dump({'timings':timings,'failures':failures},f)
    else:
        logger.info(f"Will read folder(s) {' and '.join((str(f) for f in folders))}")
        matname_to_data,matname_to_inf=read_and_analyze_folders(folders,
                    only_matload_info=only_material, only_meas_groups=only_meas_groups,cached_glob=cached_glob)
        for matname in matname_to_data:
            logger.info(f"Uploading {matname}")
            db.push_data(matname_to_inf[matname],matname_to_data[matname],
                         clear_all_from_material=clear_all_from_material, user_called=user_called)
def heal(db: PostgreSQLDatabase,force_all_meas_groups=False):
    """ Goes in order of most recent material first, and within that, reloads first than re-extractions."""
    from datavac.io.meta_reader import perform_extraction, get_cached_glob
    cached_glob=get_cached_glob()
    fullname=CONFIG['database']['materials']['full_name']
    matcolnames=[*CONFIG['database']['materials']['info_columns']]
    loadcolnames=[*CONFIG['database']['loads']['info_columns']]
    with db.engine_connect() as conn:
        res=conn.execute(select(db._rextab.c.matid,
                                *[db._mattab.c[n] for n in [fullname]+matcolnames],
                                *[db._rextab.c[n] for n in loadcolnames],
                                )\
                         .select_from(db._rextab.join(db._mattab))\
                         .order_by(db._mattab.c.date_user_changed.desc())).all()
        if not len(res):
            logger.info("Nothing needs re-loading or re-extracting!")
        else:
            for matid,matname,*other_matinfo in res:
                # Reloads
                logger.info(f"Looking at {matname}: {other_matinfo}")
                res=conn.execute(select(db._rextab.c.MeasGroup) \
                                 .where(db._rextab.c.matid==matid)\
                                 .where(db._rextab.c.full_reload==True)).all()
                if not len(res):
                    logger.info("Nothing to reload")
                else:
                    logger.info("Doing reloads")
                    meas_groups=None if force_all_meas_groups else [r[0] for r in res]
                    all_meas_groups=ensure_meas_group_sufficiency(meas_groups,on_error='ignore')
                    read_and_upload_data(db,
                         only_material=dict(**{fullname:matname},
                                            **dict(zip(matcolnames+loadcolnames,[[om] for om in other_matinfo]))),
                         only_meas_groups=all_meas_groups,
                         clear_all_from_material=False,user_called=False,cached_glob=cached_glob)

                # Re-extracts
                res=conn.execute(select(db._rextab.c.MeasGroup) \
                                 .where(db._rextab.c.matid==matid) \
                                 .where(db._rextab.c.full_reload==False)).all()
                if not len(res):
                    logger.info("Nothing to re-extract")
                else:
                    logger.info("Doing re-extractions")
                    meas_groups=[r[0] for r in res]
                    all_meas_groups=ensure_meas_group_sufficiency(meas_groups,on_error='ignore')
                    logger.info(f"Pulling sweeps for {all_meas_groups} to re-extract {meas_groups}")
                    mumts={mg:db.get_data_for_regen(mg,matname=matname,on_no_data=None,conn=conn) for mg in all_meas_groups}
                    mumts={k:v for k,v in mumts.items() if v is not None}
                    logger.info(f"Re-extracting {meas_groups}")
                    perform_extraction({matname:mumts})
                    logger.info(f"Pushing new extraction for {meas_groups}")
                    db.push_data({fullname:matname},{k:v for k,v in mumts.items() if k in meas_groups},
                                 clear_all_from_material=False,
                                 user_called=False, re_extraction=True, defer_analyses=True)

        res=conn.execute(select(db._reatab.c.matid,
                                *[db._mattab.c[n] for n in [fullname]+matcolnames],) \
                         .select_from(db._reatab.join(db._mattab)) \
                         .order_by(db._mattab.c.date_user_changed.desc())).all()
        if not len(res):
            logger.info("Nothing needs re-analyzing!")
        else:
            logger.info("Doing re-analyses")
            for matid,matname,*other_matinfo in res:
                # Re-analyses
                res=conn.execute(select(db._reatab.c.analysis) \
                                 .where(db._reatab.c.matid==matid)).all()
                if not len(res):
                    logger.info("Nothing to re-analyze")
                else:
                    logger.info(f"Looking at {matname}")
                    analyses=[r[0] for r in res]
                    req_meas_groups=CONFIG.get_dependency_meas_groups_for_analyses(analyses,required_only=True)
                    all_meas_groups=CONFIG.get_dependency_meas_groups_for_analyses(analyses,required_only=False)
                    logger.info(f"Pulling measured and extracted data for {list(dict(**all_meas_groups).keys())}")
                    mumts={}
                    from datavac.io.measurement_table import MultiUniformMeasurementTable, UniformMeasurementTable
                    precol_loadids={}
                    for mg in all_meas_groups:
                        mg_info = CONFIG['measurement_groups'][mg]
                        condie = mg_info.get('connect_to_die_table', True)
                        conlay = mg_info.get('connect_to_layout_table', True)
                        data=db.get_data(meas_group=mg, include_sweeps=False,
                                         scalar_columns=[*mg_info['meas_columns'],*mg_info['analysis_columns'],
                                                         'loadid',*(['DieXY'] if condie else []),*(['Structure'] if conlay else []),
                                                         CONFIG['database']['materials']['full_name'],
                                                         *CONFIG['database']['materials']['info_columns']],
                                         **{CONFIG['database']['materials']['full_name']:[matname]})
                        if mg in req_meas_groups:
                            #assert len(data), f"Missing required data for '{mg}' in {matname}"
                            if not len(data):
                                logger.warning(f"Missing required data for '{mg}' in {matname}")
                                continue
                        if len(data):
                            mumts[mg]=UniformMeasurementTable(dataframe=data,headers=[],
                                                              meas_length=None,meas_type=None,meas_group=mg)
                            assert len(list(data['loadid'].unique()))==1
                            precol_loadids[mg]=data['loadid'].iloc[0]
                    logger.info(f"Re-analyzing {analyses} for {matname}")
                    db.perform_analyses(conn,analyses=analyses,
                                        precollected_data_by_meas_group=mumts,
                                        precollected_loadids=precol_loadids,
                                        precollected_matid=matid)
                    conn.commit()

        logger.info(f"Done healing.")

def cli_heal(*args):
    parser=argparse.ArgumentParser(description='Tries to re-extract or re-load dumped info')
    parser.add_argument('-a','--all_measgroups',action='store_true',help="Force re-upload of all meas groups for each healed lot")
    namespace=parser.parse_args(args)
    db=get_database()
    heal(db,force_all_meas_groups=namespace.all_measgroups)

def cli_clear_reextract_list(*args):
    parser=argparse.ArgumentParser(description='Clears the list of items which will be re-extracted upon healing')
    parser.add_argument('-g','--group',action='append',help='Measurement group(s) to clear from list, eg -g GROUP1 -g GROUP2')
    namespace=parser.parse_args(args)

    self=get_database()

    dstat=delete(self._rextab)
    if namespace.group is not None:
        dstat=dstat.where(self._rextab.c.MeasGroup.in_(namespace.group))
    with self.engine_begin() as conn:
        conn.execute(dstat)

def cli_clear_reanalyze_list(*args):
    parser=argparse.ArgumentParser(description='Clears the list of items which will be re-analyzed upon healing')
    parser.add_argument('-a','--analysis',action='append',help='Analyses to clear from list, eg -a ANALYSIS1 -a ANALYSIS2')
    namespace=parser.parse_args(args)

    self=get_database()

    dstat=delete(self._reatab)
    if namespace.analysis is not None:
        dstat=dstat.where(self._reatab.c.analysis.in_(namespace.analysis))
    with self.engine_begin() as conn:
        conn.execute(dstat)

def cli_print_database(*args):
    parser=argparse.ArgumentParser(description='Prints the database connection info')
    namespace=parser.parse_args(args)
    print({k:v for k,v in get_db_connection_info().items() if k!='Password'})
def cli_update_mask_info(*args):
    parser=argparse.ArgumentParser(description='Updates the mask information in the database')
    namespace=parser.parse_args(args)
    self=get_database(metadata_source='reflect')
    with self.engine_connect() as conn:
        self.update_mask_info(conn)
        conn.commit()
def cli_upload_all_data(*args):
    parser=argparse.ArgumentParser(description='Reads EVERYTHING, folder by folder, and outputs timing and successes')
    folders=sorted([f.stem for f in Path(os.environ['DATAVACUUM_READ_DIR']).glob('*')])
    timings={}
    failures={}
    all_start_time=time.time()
    for folder in folders:
        start_time=time.time()
        try:
            print(folder)
            cli_upload_data('-f',folder)
        except Exception as e:
            logger.critical(f"Failed on {folder}")
            logger.critical(str(e))
            failures[folder]=str(e)
        end_time=time.time()
        timings[folder]=end_time-start_time
    all_end_time=time.time()
    print("##############################################")
    print(f"Complete in {(all_end_time-all_start_time)/60/60} hours")
    print("##### Timings")
    for f,t in timings.items():
        print(f"{f}: {t} seconds")
    print("##### Failures")
    for f,fail in failures.items():
        print(f"{f}: {fail}")
    with open("upload_all.pkl",'wb') as f:
        pickle.dump({'timings':timings,'failures':failures},f)



class DDFDatabase(Database):
    def __init__(self,ddf={}):
        self._ddf=ddf
    def get_data(self,meas_group,scalar_columns=None,include_sweeps=False,unstack_headers=False,raw_only=False,**factors):
        assert unstack_headers
        assert not raw_only

        df=self._ddf[meas_group]
        avail_header_cols=[k for k,v in df.dtypes.items() if str(v)=='object']
        avail_scalar_cols=[k for k in df.columns if k not in avail_header_cols]

        df=df[functools.reduce(np.logical_and,
                     [df[fname].isin(fvals) for fname,fvals in factors.items()],
                     pd.Series([True]*len(df)))]

        cols=[*(avail_header_cols if include_sweeps is True else include_sweeps if include_sweeps else []),
              *(scalar_columns if scalar_columns else avail_scalar_cols)]
        return df[cols].reset_index()
    def get_factors(self,meas_group,factor_names,pre_filters={}):
        df=self._ddf[meas_group]
        df=df[functools.reduce(np.logical_and,
                     [df[fname].isin(fvals) for fname,fvals in pre_filters.items()],
                     pd.Series([True]*len(df)))]
        return {fn:list(df[fn].unique()) for fn in factor_names}


def pickle_db_cached(namer: Union[Callable,str], namespace:Optional[str]=None, conn:Connection=None):
    def wrapper(func):
        cache_dir=USER_CACHE/namespace
        if not cache_dir.exists(): cache_dir.mkdir()

        @functools.wraps(func)
        def wrapped(*args,force=False,**kwargs):
            name=namer if type(namer) is str else namer(*args,**kwargs)
            cfile=cache_dir/name
            blob_name=f'{namespace}.{name}'
            if not force:
                with time_it(f"Get DB cache time for {name}",threshold_time=.005):
                    try: db_cached_time:float= get_blob_store().get_obj_date(blob_name,conn=conn).timestamp()
                    # If it's an SQLAlchemyError, the transaction might be aborted so have to raise this
                    # Silently aborted transations could cause some downstream use of this connection to fail
                    except SQLAlchemyError as e: raise e
                    # Otherwise, we'll just try to fix it
                    except Exception as e:
                        logger.debug("Couldn't get DB cache time: "+str(e));
                        db_cached_time:float= -np.inf
                with time_it(f"Get local cache time for {name}",threshold_time=.005):
                    try: lc_cached_time=cfile.stat().st_mtime
                    except Exception as e:
                        logger.debug("Couldn't get local cache time: "+str(e));
                        lc_cached_time:float= -np.inf
                if (not np.isfinite(db_cached_time)) and (not np.isfinite(lc_cached_time)):
                    logger.debug(f"Couldn't get local or DB cache time for {name}")
                    force=True
            if not force:
                if db_cached_time<lc_cached_time:
                    try:
                        with time_it(f"Reading local cache for {name}",threshold_time=.005):
                            with open(cfile,'rb') as f: return pickle.load(f),lc_cached_time
                    except Exception as e:
                        logger.debug(f"Couldn't read local cache for {name}: {str(e)}")
                try:
                    with time_it(f"Reading DB cache for {name}",threshold_time=.025):
                        res,cached_time=get_blob_store().get_obj(f'{namespace}.{name}',conn=conn),db_cached_time
                except Exception as e:
                    logger.debug(f"Couldn't read DB cache for {name}: {str(e)}")
                    force=True
                else:
                    with time_it(f"Storing local cache for {name}",threshold_time=.005):
                        with open(cfile,'wb') as f: pickle.dump(res,f)
                    return res,cached_time
            with time_it(f"Generating {name}"):
                res=func(*args,**kwargs)

            with time_it(f"Storing DB cache for {name}",threshold_time=.025):
                get_blob_store().store_obj(blob_name,res,conn=conn)
            with time_it(f"Storing local cache for {name}",threshold_time=.005):
                with open(cfile,'wb') as f: pickle.dump(res,f)
            return res,datetime.now().timestamp()
        return wrapped
    return wrapper


def cli_test_db_connection_speed(*args):
    parser=argparse.ArgumentParser(description='Tests the speed of the database connection')
    parser.add_argument('-n','--num',type=int,default=1000,help='Number of times to test')
    namespace=parser.parse_args(args)
    db=get_database()
    times=[]
    with db.engine_connect() as conn:
        with time_it(f"Testing DB connection speed {namespace.num} times",threshold_time=.1):
            for i in range(namespace.num):
                start_time=time.time()
                conn.execute(select(1))
                end_time=time.time()
                times.append(end_time-start_time)
    print(f"Mean time: {np.mean(times)}")
    print(f"Med  time: {np.median(times)}")
    print(f"Max  time: {np.max(times)}")
    print(f"Min  time: {np.min(times)}")
    print(f"Std  time: {np.std(times)}")
    print("Done")

def cli_dump_material(*args):
    parser=argparse.ArgumentParser(description='Dumps a material from the database')
    parser.add_argument("-g","--group",action='append',help="Restrict to specified measurement group(s): " \
                                                              "eg -g GROUP1 -g GROUP2")
    # For the moment, only full matname allowed... and required
    ALL_MATERIAL_COLUMNS=[CONFIG['database']['materials']['full_name']]

    for col in ALL_MATERIAL_COLUMNS:
        parser.add_argument(f'--{col.lower()}',action='store',
                            help=f"Restrict to specified {col}(s): " \
                                 f"eg --{col.lower()} {col.upper()}1 --{col.lower()} {col.upper()}2")


    namespace=parser.parse_args(args)
    assert getattr(namespace,CONFIG['database']['materials']['full_name'].lower()) is not None, "For now, need to supply FULL MATERIAL IDENTIFIER"
    only_material={col: getattr(namespace,col.lower()) for col in ALL_MATERIAL_COLUMNS}
    only_material={k:v for k,v in only_material.items() if v is not None}

    assert namespace.group is None or len(namespace.group)
    db=get_database()
    with db.engine_begin() as conn:
        db.dump_material(only_material,conn,only_meas_group=namespace.group)

cli_database=cli_helper(cli_funcs={
    'clear': 'datavac.io.database:cli_clear_database',
    'upload_data (ud)': 'datavac.io.database:cli_upload_data',
    'upload_all_data': 'datavac.io.database:cli_upload_all_data',
    'dump_extraction': 'datavac.io.database:cli_dump_extraction',
    'dump_measurement': 'datavac.io.database:cli_dump_measurement',
    'dump_analysis': 'datavac.io.database:cli_dump_analysis',
    'dump_material': 'datavac.io.database:cli_dump_material',
    'print': 'datavac.io.database:cli_print_database',
    'clear_reextract_list': 'datavac.io.database:cli_clear_reextract_list',
    'clear_reanalyze_list': 'datavac.io.database:cli_clear_reanalyze_list',
    'force': 'datavac.io.database:cli_force_database',
    'update_mask_info': 'datavac.io.database:cli_update_mask_info',
    'heal': 'datavac.io.database:cli_heal',
    'test_db_connect':'datavac.io.database:cli_test_db_connection_speed',
})