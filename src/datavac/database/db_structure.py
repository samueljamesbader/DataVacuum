from typing import TypedDict
from datavac.util.util import only
from sqlalchemy import BOOLEAN, DOUBLE_PRECISION, ForeignKeyConstraint, Table, Column, ForeignKey,\
    PrimaryKeyConstraint, INTEGER, VARCHAR, MetaData, TypeEngine, UniqueConstraint
from sqlalchemy.dialects.postgresql import BYTEA, TIMESTAMP

from datavac.config.data_definition import DataDefinition
from datavac.config.project_config import PCONF

# Singleton for the database structure.
_dbs: 'DBStructure | None' = None
def DBSTRUCT() -> 'DBStructure':
    """Returns the database structure singleton."""
    global _dbs
    if _dbs is None: _dbs = DBStructure()
    return _dbs

# Default mappings between pandas and SQLAlchemy types.
pd_to_sql_types: dict[str,TypeEngine]={
    'int64':INTEGER,'Int64':INTEGER,
    'int32':INTEGER,'Int32':INTEGER,
    'string':VARCHAR,
    'str':VARCHAR,
    'float32':DOUBLE_PRECISION,'Float32':DOUBLE_PRECISION,
    'float64':DOUBLE_PRECISION,'Float64':DOUBLE_PRECISION,
    'bool':BOOLEAN,'boolean':BOOLEAN,
    'datetime64[ns]':TIMESTAMP, 'object':BYTEA}
sql_to_pd_types: dict[TypeEngine, str]={
    INTEGER:'Int32',
    VARCHAR:'string',
    BOOLEAN: 'bool',
    DOUBLE_PRECISION:'float64',
    TIMESTAMP:'datetime64[ns]'
}

# Just a convenience for the number of times these are used as kwargs
class _CASCTD(TypedDict):
    onupdate: str
    ondelete: str
_CASC: _CASCTD=_CASCTD(onupdate='CASCADE',ondelete='CASCADE')

class DBStructure():
    
    metadata: MetaData 
    datadef: DataDefinition
    int_schema: str = 'vac'

    
    def __init__(self):
        self.metadata = MetaData()
        self.datadef = PCONF().data_definition

    def get_sample_dbtable(self) -> Table:
        """Returns an SQLAlchemy Table object for the sample info."""
        sample_identifier = self.datadef.sample_identifier_column
        return self.metadata.tables.get(self.int_schema + '.Samples') or \
            Table('Samples', self.metadata,
                  Column('sampleid', INTEGER, primary_key=True, autoincrement=True),
                  Column(sample_identifier.name,sample_identifier.sql_dtype,unique=True, nullable=False),
                  *[Column(sd.key_column.name, sd.key_column.sql_dtype,
                           ForeignKey(self.get_sample_descriptor_dbtable(sd.name).c[sd.key_column.name], **_CASC),
                           nullable=False)
                        for sd_name in self.datadef.sample_reference_names()
                            for sd in [self.datadef.sample_reference(sd_name)]],
                  *[Column(c.name, c.sql_dtype, nullable=False) for c in self.datadef.sample_info_columns])

    def get_trove_load_dbtable(self, trove_name: str) -> Table:
        """Returns an SQLAlchemy Table object for the given trove's Loads table."""
        trove = self.datadef.trove(trove_name)
        return self.metadata.tables.get(self.int_schema+f'.Loads_{trove_name}') or \
            Table('Loads_{trove_name}', self.metadata,
                  Column('loadid',INTEGER,primary_key=True,autoincrement=True),
                  Column('matid',INTEGER,ForeignKey(self.get_sample_dbtable().c.matid,**_CASC),nullable=False),
                  Column('MeasGroup',VARCHAR,nullable=False),
                  *[Column(c.name,c.sql_dtype,nullable=False) for c in trove.load_info_columns],
                  UniqueConstraint('matid','MeasGroup'),
                  schema=self.int_schema)
    
    def get_sample_reference_dbtable(self, sr_name: str) -> Table:
        """Returns an SQLAlchemy Table object for the given sample reference."""
        sr = self.datadef.sample_reference(sr_name)
        return self.metadata.tables.get(self.int_schema + f'.{sr.name}') or \
            Table(sr.name, self.metadata,
                  Column(sr.key_column.name, sr.key_column.sql_dtype,
                         primary_key=True, nullable=False),
                  *[Column(c.name, c.sql_dtype, nullable=False) for c in sr.info_columns],
                  schema=self.int_schema)
    
    def get_subsample_reference_dbtable(self, ssr_name: str) -> Table:
        """Returns an SQLAlchemy Table object for the given subsample reference."""
        ssr = self.datadef.subsample_reference(ssr_name)
        return self.metadata.tables.get(self.int_schema + f'.{ssr.name}') or \
            Table(ssr.name, self.metadata,
                  Column(ssr.key_column.name, ssr.key_column.sql_dtype, primary_key=True, nullable=False),
                  *[Column(c.name, c.sql_dtype, nullable=False) for c in ssr.info_columns],
                  schema=self.int_schema)
    
    def get_sample_descriptor_dbtable(self, sd_name: str) -> Table:
        """Returns an SQLAlchemy Table object for the given sample descriptor."""
        sd = self.datadef.sample_descriptor(sd_name)
        return self.metadata.tables.get(self.int_schema + f'.{sd.name}') or \
            Table(sd.name, self.metadata,
                  Column('sampleid', INTEGER, ForeignKey(self.get_sample_dbtable().c.sampleid, **_CASC), nullable=False),
                    *[Column(c.name, c.sql_dtype, nullable=False) for c in sd.info_columns],
                    schema=self.int_schema)
                    
    def get_measurement_group_dbtables(self, mg_name: str) -> dict[str,Table]:
        """Returns a dictionary of SQLAlchemy Table objects for the given measurement group."""

        mg=self.datadef.measurement_group(mg_name)
        trove_name= only(mg.reader_cards.keys())

        meas_tab = self.metadata.tables.get(self.int_schema+f'.Meas -- {mg_name}') or \
            Table(
                f'Meas -- {mg_name}', self.metadata,
                Column('loadid',INTEGER,ForeignKey(self.get_trove_load_dbtable(trove_name).c.loadid,**_CASC),nullable=False),
                Column('measid',INTEGER,nullable=False),
                *[Column(sd.key_column.name, sd.key_column.sql_dtype,
                         ForeignKey(f'{sd.name}.{sd.key_column.name}',name=f'fk_{sd.name}',**_CASC),nullable=False)
                         for sd_name in mg.subsample_reference_names for sd in [self.datadef.subsample_reference(sd_name)]],
                Column('rawgroup',INTEGER,nullable=False),
                *[Column(c.name,c.sql_dtype) for c in mg.meas_columns],
                PrimaryKeyConstraint('loadid','measid'),
                schema=self.int_schema)
        extr_tab = self.metadata.tables.get(self.int_schema+f'.Extr -- {mg_name}') or \
            Table(
                f'Extr -- {mg_name}', self.metadata,
                Column('loadid',INTEGER,nullable=False),
                Column('measid',INTEGER,nullable=False),
                ForeignKeyConstraint(columns=['loadid','measid'],**_CASC,
                                     refcolumns=[meas_tab.c.loadid, meas_tab.c.measid]),
                *[Column(c.name, c.sql_dtype) for c in mg.extr_columns])
        
        sweep_tab = self.metadata.tables.get(self.int_schema+f'.Sweep -- {mg_name}') or \
            Table(
                f'Sweep -- {mg_name}', self.metadata,
                Column('loadid', INTEGER, nullable=False),
                Column('measid', INTEGER, nullable=False),
                Column('sweep', BYTEA, nullable=False),
                Column('header', VARCHAR, nullable=False),
                PrimaryKeyConstraint('loadid', 'measid', 'header'),
                ForeignKeyConstraint(columns=['loadid', 'measid'], **_CASC,
                                     refcolumns=[meas_tab.c.loadid, meas_tab.c.measid]))

        return {'meas': meas_tab, 'extr': extr_tab, 'sweep': sweep_tab}
