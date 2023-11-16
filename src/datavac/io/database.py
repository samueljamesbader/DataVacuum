import functools
import os
from pathlib import Path

import sqlalchemy
from sqlalchemy import text, MetaData, Engine, create_engine
from sqlalchemy import Table, Integer, Numeric, String, Boolean, Column
from sqlalchemy import Table, MetaData
import numpy as np
from sqlalchemy.engine import URL
import io
import pandas as pd

from datavac.logging import logger, time_it
from datavac.io.measurement_table import MeasurementTable


class Database:

    _sync_engine:  Engine = None
    CERT_DIR = Path(os.environ['DATAVACUUM_CERT_DIR'])

    def __init__(self):
        self._alchemy_table_cache = {}

    def get_engine(self):
        if self._sync_engine is None:
            #with time_it(f"Getting engine"):
                self._drivername=os.environ["DATAVACUUM_DB_DRIVERNAME"]
                if self._drivername=='postgresql':
                    connection_info=dict([[s.strip() for s in x.split("=")] for x in
                                          os.environ['DATAVACUUM_DBSTRING'].split(";")])
                    url=URL.create(
                        drivername='postgresql',
                        username=connection_info['Uid'],
                        password=connection_info['Password'],
                        host=connection_info['Server'],
                        port=int(connection_info['Port']),
                        database=connection_info['Database'],
                    )
                    # For synchronous
                    if connection_info['Server']=='localhost':
                        ssl_args={}
                    else:
                        ssl_args = {'sslmode':'verify-full',
                                    'sslrootcert': str(self.CERT_DIR/'IntelSHA256RootCA-Base64.crt')}
                    self._sync_engine=create_engine(url, connect_args=ssl_args, pool_recycle=60)
                elif self._drivername=='sqlite':
                    folder:Path=Path(os.environ['DATAVACUUM_CACHE_DIR'])/"db"
                    assert "//" not in str(folder) and "\\\\" not in str(folder), \
                        f"DATAVACUUM_CACHE_DIR points to a remote directory [{os.environ['DATAVACUUM_CACHE_DIR']}].  " \
                        "This would be miserably slow to use for SQLITE."
                    folder.mkdir(exist_ok=True)
                    self._sync_engine=create_engine(f"sqlite:///{folder}/SummaryData.db")
        return self._sync_engine

    @staticmethod
    def to_alchemy_columns(table: pd.DataFrame):
        if 'object' in [str(x) for x in table.dtypes]:
            import pdb; pdb.set_trace()
        type_mapping={
            'int64':Integer,'Int64':Integer,
            'int32':Integer,'Int32':Integer,
            'string':String,
            'float32':Numeric(asdecimal=False),'Float32':Numeric(asdecimal=False),
            'float64':Numeric(asdecimal=False),'Float64':Numeric(asdecimal=False),
            'bool':Boolean,'boolean':Boolean}
        return [Column(c,type_mapping[str(cdtype)],primary_key=(c=='LotWafer')) for c,cdtype in table.dtypes.items()]

    @staticmethod
    def dtypes_from_alchemy_table(tab:Table):
        type_mapping={
            'INTEGER':'Int32',
            'VARCHAR':'string',
            'NUMERIC':'Float64',
            'BOOLEAN':'boolean'}
        return {c.name:type_mapping[str(c.type)] for c in tab.columns}

    def make_summary_tables_from_proto(self,source_name,analysis_name,tables,only_meas_groups=[]):
        for meas_group in only_meas_groups:
            if meas_group not in tables: continue
            table=tables[meas_group]
            table_name=f'{source_name}--{analysis_name}--{meas_group}--Summary'
            if 'in_place' in table.columns:
                raise Exception("There's a column in the table called 'in_place'; that seems like an accident...")

            #with engine.begin() as conn:
            #    table.replace([np.inf,-np.inf],np.nan).iloc[:0].to_sql(table_name,conn.connection,index=False,if_exists='replace')
            #    conn.connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN \"MeasID\" BIGSERIAL;"))
            metadata=MetaData()
            engine=self.get_engine()
            with engine.begin() as conn:
                sa_table=Table(table_name,metadata,Column('MeasIndex',Integer,primary_key=True),*self.to_alchemy_columns(table))
                if sqlalchemy.inspect(engine).has_table(table_name):
                    logger.debug(f"Table '{table_name}' exists, deleting and recreating")
                    self.get_alchemy_table.cache_clear()
                    sa_table.drop(conn)
                sa_table.create(conn)

    def upload_summary(self,source_name,lot,wafer,analysis_name,tables:dict[str,MeasurementTable],only_meas_groups=None,allow_delete_table=False):
        logger.debug(f"In Upload Summary with allow_delete_table={allow_delete_table}")
        for meas_group in (only_meas_groups if only_meas_groups else tables.keys()):
            if meas_group not in tables: continue
            if isinstance(tables[meas_group],MeasurementTable):
                logger.warning(f"Uploading {meas_group} with layout params, not intended")
                table=tables[meas_group].scalar_table_with_layout_params()
                #import pdb; pdb.set_trace()
            elif isinstance(tables[meas_group],pd.DataFrame):
                table=tables[meas_group]
                if type(table.columns[0]) is not tuple:
                    logger.warning(f"Uploading {meas_group} from DataFrame")
                else:
                    continue
                #import pdb; pdb.set_trace()
            else:
                continue
            if 'MeasIndex' in table:
                assert np.all(table['MeasIndex']==table.index)
                table=table.drop(columns=['MeasIndex'])
            table_name=f'{source_name}--{analysis_name}--{meas_group}--Summary'

            insp = sqlalchemy.inspect(self.get_engine())
            table_exists=insp.has_table(table_name)
            if table_exists:
                old_table_columns=[c.name for c in self.get_alchemy_table(table_name).columns]
                new_table_columns=['MeasIndex']+[c for c in table.columns]
                if new_table_columns!=old_table_columns:
                    logger.warning(f"DB    Table Columns: {old_table_columns}")
                    logger.warning(f"Local Table Columns: {new_table_columns}")
                    if allow_delete_table is False:
                        raise Exception("Your data has a different set of columns than existing data" \
                                        " so it can't be uploaded!  Please alert Sam.")
                    else:
                        self.get_alchemy_table.cache_clear()
                        #raise Exception("Something seems to go wrong for the first wafer of a new one... please slow down and debug here next time")
                    logger.warning(f"Table {table_name} has wrong columns, deleting it all")
                    self.make_summary_tables_from_proto(source_name, analysis_name, {meas_group:table}, only_meas_groups=[meas_group])

            if not table_exists:
                logger.warning(f"Table {table_name} doesn't exist yet, creating")
                logger.warning(f"Local table columns are {['MeasIndex']+[c for c in table.columns]} ")
                self.make_summary_tables_from_proto(source_name, analysis_name, {meas_group:table}, only_meas_groups=[meas_group])
            if self._drivername=='postgresql':
                self._fast_postgresql_upload(table_name,table,lot,wafer)
            elif self._drivername=='sqlite':
                #print("Pausing before uploading",table_name)
                #import pdb;pdb.set_trace()
                self._sqlite_upload(table_name,table,lot,wafer)

    def _sqlite_upload(self,table_name,table: pd.DataFrame,lot,wafer):
            connection=self.get_engine().connect()
            try:
                connection.execute(text(f'DELETE from "{table_name}" WHERE "LotWafer"=:lotwafer'),{'lotwafer':f"{lot}_{wafer}"})
                #table.to_sql(table_name,connection,if_exists='append')
                insert_query=self.get_alchemy_table(table_name).insert()
                df=table.reset_index().rename(columns={'index':'MeasIndex'})
                #for k in df.columns:
                #    if str(df[k].dtype) in ['Float64','Float32']:
                #        df[k]=df[k].astype('float32')
                #    if str(df[k].dtype) in ['boolean']:
                #        df[k]=df[k].astype('object').fillna(value=None)
                #import pdb; pdb.set_trace()
                connection.execute(insert_query,[{k:(None if pd.isna(v) else v) for k,v in rec.items()} for rec in df.to_dict(orient='records')])
                connection.commit()
            finally:
                connection.close()


    def _fast_postgresql_upload(self,table_name,table,lot,wafer):
            connection=self.get_engine().connect()
            try:
                if lot is not None and wafer is not None:
                    connection.execute(text(f'DELETE from "{table_name}" WHERE "LotWafer"=:lotwafer'),
                                       {'lotwafer':f"{lot}_{wafer}"})

                # Ge the underlying DB-API connection because this is postgresql-specific
                conn=connection.connection

                #stream the data using 'to_csv' and StringIO()
                output = io.StringIO()
                table.replace([np.inf,-np.inf],np.nan).to_csv(output, sep='|', header=False, index=True)

                #jump to start of stream
                output.seek(0)
                #contents = output.getvalue()

                # Use postgresql's 'copy_from' function to load that csv stream into the DB
                with conn.cursor() as cur:
                    #null values become ''
                    try:
                        cur.copy_from(output, table_name, null="", sep='|')
                    except Exception as e:
                        logger.critical(f"Local table columns are {['MeasIndex']+[c for c in table.columns]} ")
                        logger.critical(f"SQL   table columns are {[c.name for c in self.get_alchemy_table(table_name).columns]} ")
                        import pdb; pdb.set_trace()

                        raise
                    conn.commit()
            finally:
                connection.close()

    @functools.cache
    def get_alchemy_table(self,table_name):
        with time_it("Inspecting alchemy table"):
            # for synchronous
            tab=Table(table_name, MetaData(), autoload_with=self.get_engine())
            for col in list(tab.columns):
                if str(col.type)=='NUMERIC':
                    col.type.asdecimal=False
            self._alchemy_table_cache[table_name]=tab
            return tab

    def make_DSN(self,rootcertfile,dsnfile_override=None):
        connection_info=dict([[s.strip() for s in x.split("=")] for x in
                              os.environ['DATAVACUUM_DBSTRING'].split(";")])
        if rootcertfile: escrootfile=str(rootcertfile).replace('\\','\\\\')
        string=\
            f"""
            [ODBC]
            DRIVER=PostgreSQL Unicode(x64)
            UID={connection_info['Uid']}
            XaOpt=1
            FetchRefcursors=0
            OptionalErrors=0
            D6=-101
            {f'pqopt={{sslrootcert={escrootfile}}}' if connection_info['Server']!='localhost' else ''}
            LowerCaseIdentifier=0
            UseServerSidePrepare=1
            ByteaAsLongVarBinary=1
            BI=0
            TrueIsMinus1=0
            UpdatableCursors=1
            LFConversion=1
            ExtraSysTablePrefixes=
            Parse=0
            BoolsAsChar=1
            UnknownsAsLongVarchar=0
            TextAsLongVarchar=1
            UseDeclareFetch=0
            CommLog=0
            Debug=0
            MaxLongVarcharSize=8190
            MaxVarcharSize=255
            UnknownSizes=0
            Fetch=100
            ShowSystemTables=0
            RowVersioning=0
            ShowOidColumn=0
            FakeOidIndex=0
            Protocol=7.4
            ReadOnly=0
            {f'SSLmode=verify-full' if connection_info['Server']!='localhost' else ''}
            PORT={connection_info['Port']}
            SERVER={connection_info['Server']}
            DATABASE={connection_info['Database']}
            """
        dsnfile=self.CERT_DIR/f"datavacuum_database.dsn" if dsnfile_override is None else dsnfile_override
        with open(dsnfile,'w') as f:
            f.write("\n".join([l.strip() for l in string.split("\n") if l.strip()!=""]))
        return dsnfile

    def make_JMPstart(self,rootcertfile,jslfile_override=None):
        connection_info=dict([[s.strip() for s in x.split("=")] for x in
                              os.environ['DATAVACUUM_DBSTRING'].split(";")])
        if rootcertfile: escrootfile=str(rootcertfile).replace('\\','\\\\')
        string=\
            f"""
            New SQL Query(
                Connection(
                    "ODBC:DRIVER={{PostgreSQL Unicode(x64)}};
                    DATABASE={connection_info['Database']};
                    SERVER={connection_info['Server']};
                    PORT={connection_info['Port']};
                    UID={connection_info['Uid']};
                    PWD={connection_info['Password']};
                    {'SSLmode=verify-full;' if connection_info['Server']!='localhost' else ''}
                    ReadOnly=0;Protocol=7.4;FakeOidIndex=0;ShowOidColumn=0;RowVersioning=0;
                    ShowSystemTables=0;Fetch=100;UnknownSizes=0;MaxVarcharSize=255;MaxLongVarcharSize=8190;
                    Debug=0;CommLog=0;UseDeclareFetch=0;TextAsLongVarchar=1;UnknownsAsLongVarchar=0;BoolsAsChar=1;
                    Parse=0;LFConversion=1;UpdatableCursors=1;TrueIsMinus1=0;BI=0;ByteaAsLongVarBinary=1;
                    UseServerSidePrepare=1;LowerCaseIdentifier=0;
                    {f'pqopt={{sslrootcert={escrootfile}}};' if connection_info['Server']!='localhost' else '' }
                    D6=-101;OptionalErrors=0;FetchRefcursors=0;XaOpt=1;"
                ),
                QueryName( "test_query" ),
                CustomSQL("Select * from information_schema.tables;"),
                PostQueryScript( "Close(Data Table(\!"test_query\!"), No Save);" )
                ) << Run;
                Print("Wait five seconds and check the connections.");
            """
        jslfile=self.CERT_DIR/f"datavacuum_database.dsn" if jslfile_override is None else jslfile_override
        with open(jslfile,'w') as f:
            f.write("\n".join([l.strip() for l in string.split("\n") if l.strip()!=""]))
        return jslfile
