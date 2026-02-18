from typing import TypedDict
from datavac.util.util import only
from sqlalchemy import BOOLEAN, DOUBLE_PRECISION, ForeignKeyConstraint, Table, Column, ForeignKey,\
    PrimaryKeyConstraint, INTEGER, VARCHAR, MetaData, UniqueConstraint
from sqlalchemy.types import TypeEngine
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
pd_to_sql_types: dict[str,TypeEngine]={ # type: ignore
    'int64':INTEGER,'Int64':INTEGER,
    'int32':INTEGER,'Int32':INTEGER, 'int':INTEGER,
    'string':VARCHAR,
    'str':VARCHAR,'float':DOUBLE_PRECISION,
    'float32':DOUBLE_PRECISION,'Float32':DOUBLE_PRECISION,
    'float64':DOUBLE_PRECISION,'Float64':DOUBLE_PRECISION,
    'bool':BOOLEAN,'boolean':BOOLEAN,
    'datetime64[ns]':TIMESTAMP, 'object':BYTEA}
sql_to_pd_types: dict[TypeEngine, str]={ # type: ignore
    INTEGER:'Int32',
    VARCHAR:'string',
    BOOLEAN: 'boolean',
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
    jmp_schema: str = 'jmp'

    
    def __init__(self):
        self.metadata = MetaData()#schema=self.int_schema)
        self.datadef = PCONF().data_definition

    def get_sample_dbtable(self) -> Table:
        """Returns an SQLAlchemy Table object for the sample info."""
        sample_identifier = self.datadef.sample_identifier_column
        t= self.metadata.tables.get(self.int_schema + '.Samples')
        return t if (t is not None) else \
            Table('Samples', self.metadata,
                  Column('sampleid', INTEGER, primary_key=True, autoincrement=True),
                  Column(sample_identifier.name,sample_identifier.sql_dtype,unique=True, nullable=False),
                  *[Column(sr.key_column.name, sr.key_column.sql_dtype,
                           ForeignKey(self.get_sample_reference_dbtable(sr_name).c[sr.key_column.name], **_CASC),
                           nullable=False)
                        for sr_name, sr in self.datadef.sample_references.items()],
                  *[Column(c.name, c.sql_dtype, nullable=False) for c in self.datadef.sample_info_columns],
                  schema=self.int_schema)

    def get_trove_dbtables(self, trove_name: str, exclude_system:bool=False) -> dict[str,Table]:
        """Returns an SQLAlchemy Table object for the given trove's Loads table."""
        trove = self.datadef.troves[trove_name]
        t= self.metadata.tables.get(self.int_schema+f'.Loads_{trove_name}')
        load_tab= t if (t is not None) else \
            Table(f'Loads_{trove_name}', self.metadata,
                  Column('loadid',INTEGER,primary_key=True,autoincrement=True),
                  Column('sampleid',INTEGER,ForeignKey(self.get_sample_dbtable().c.sampleid,**_CASC),nullable=False),
                  Column('MeasGroup',VARCHAR,nullable=False),
                  *[Column(c.name,c.sql_dtype,nullable=False) for c in trove.load_info_columns],
                  UniqueConstraint('sampleid','MeasGroup'),
                  schema=self.int_schema)
        t= self.metadata.tables.get(self.int_schema+f'.ReLoad_{trove_name}')
        reload_tab= t if (t is not None) else \
            Table(f'ReLoad_{trove_name}', self.metadata,
                    Column('sampleid',INTEGER,ForeignKey(self.get_sample_dbtable().c.sampleid,**_CASC),nullable=False),
                    Column('MeasGroup',VARCHAR,nullable=False),
                    *[Column(c.name,c.sql_dtype,nullable=False) for c in trove.load_info_columns],
                    UniqueConstraint('sampleid','MeasGroup'),
                    schema=self.int_schema)
        t= self.metadata.tables.get(self.int_schema+f'.ReExtr_{trove_name}')
        reextr_tab= t if (t is not None) else \
            Table(f'ReExtr_{trove_name}', self.metadata,
                    Column('loadid', INTEGER, ForeignKey(load_tab.c.loadid, **_CASC), nullable=False),
                    #Column('MeasGroup',VARCHAR,primary_key=True,nullable=False),
                    schema=self.int_schema)
        att_data,att_syst=trove.additional_tables(int_schema=self.int_schema, metadata=self.metadata, load_tab=load_tab)
        if exclude_system:
            return {'loads': load_tab, **att_data}
        else:
            return {'loads': load_tab, 'reload': reload_tab, 'reextr': reextr_tab, **att_data, **att_syst}
                    
    
    def get_sample_reference_dbtable(self, sr_name: str) -> Table:
        """Returns an SQLAlchemy Table object for the given sample reference."""
        sr = self.datadef.sample_references[sr_name]
        t= self.metadata.tables.get(self.int_schema + f'.{sr.name}')
        return t if (t is not None) else \
            Table(sr.name, self.metadata,
                  Column(sr.key_column.name, sr.key_column.sql_dtype,
                         primary_key=True, nullable=False),
                  *[Column(c.name, c.sql_dtype, nullable=False) for c in sr.info_columns],
                  schema=self.int_schema)
    
    def get_subsample_reference_dbtable(self, ssr_name: str) -> Table:
        """Returns an SQLAlchemy Table object for the given subsample reference."""
        ssr = self.datadef.subsample_references[ssr_name]
        t= self.metadata.tables.get(self.int_schema + f'.{ssr.name}')
        return t if (t is not None) else \
            Table(ssr.name, self.metadata,
                  Column(ssr.key_column.name, ssr.key_column.sql_dtype, primary_key=True, nullable=False),
                  *[Column(c.name, c.sql_dtype, nullable=True) for c in ssr.info_columns],
                  *ssr.get_constraints(),
                  schema=self.int_schema)
    
    def get_sample_descriptor_dbtable(self, sd_name: str) -> Table:
        """Returns an SQLAlchemy Table object for the given sample descriptor."""
        sd = self.datadef.sample_descriptors[sd_name]
        t= self.metadata.tables.get(self.int_schema + f'.{sd.name}')
        return t if (t is not None) else \
            Table(sd.name, self.metadata,
                  Column('sampleid', INTEGER, ForeignKey(self.get_sample_dbtable().c.sampleid, **_CASC), nullable=False, primary_key=True),
                    *[Column(c.name, c.sql_dtype, nullable=True) for c in sd.info_columns],
                    schema=self.int_schema)
    
    def get_measurement_group_dbtables(self, mg_name: str) -> dict[str,Table]:
        """Returns a dictionary of SQLAlchemy Table objects for the given measurement group."""

        mg=self.datadef.measurement_groups[mg_name]

        meas_tab = self.metadata.tables.get(self.int_schema+f'.Meas -- {mg_name}')
        if meas_tab is None:
            trove_name= only(mg.reader_cards.keys())
            try:
                subsample_references={ssr_name: self.datadef.subsample_references[ssr_name]
                                        for ssr_name in mg.subsample_reference_names}
            except KeyError as e:
                raise KeyError(f"Measurement group '{mg_name}' requested subsample references {mg.subsample_reference_names}, "
                               f"but only the following are available: {list(self.datadef.subsample_references.keys())}, "
                               f"errored on {str(e)}") from e
            subsample_reference_columns = \
                [Column(ssr.key_column.name, ssr.key_column.sql_dtype,
                             ForeignKey(self.get_subsample_reference_dbtable(ssr_name).c[ssr.key_column.name],
                                        name=f'fk_{ssr.key_column.name} -- {mg_name}',**_CASC),nullable=False)
                         for ssr_name, ssr in subsample_references.items()]
            trove_reference_columns =\
                [Column(c.name, c.sql_dtype,
                         ForeignKey(self.get_trove_dbtables(trove_name)[trove_tab_name].c[c.name],
                                    name=f'fk_{c.name} -- {mg_name}',**_CASC),nullable=False)
                         for c,trove_tab_name in mg.trove().trove_reference_columns().items()]
            print("############################")
            print(trove_reference_columns)
            meas_tab=Table(
                    f'Meas -- {mg_name}', self.metadata,
                    Column('loadid',INTEGER,ForeignKey(self.get_trove_dbtables(trove_name)['loads'].c.loadid,**_CASC),nullable=False),
                    Column('measid',INTEGER,nullable=False),
                    *subsample_reference_columns,
                    *trove_reference_columns,
                    Column('rawgroup',INTEGER,nullable=False),
                    *[Column(c.name,c.sql_dtype) for c in mg.meas_columns],
                    PrimaryKeyConstraint('loadid','measid'),
                    schema=self.int_schema)
            
        # TODO: REMOVE this temporary check
        if hasattr(mg, 'analyze'): raise Exception(f"The analyze method (in {mg.__class__}) is *hard-deprecated* for MeasurementGroup.")

        extr_tab = self.metadata.tables.get(self.int_schema+f'.Extr -- {mg_name}')
        if extr_tab is None:
            avail=mg.available_extr_columns()
            extr_column_names= mg.extr_column_names
            try:
                extr_columns = [Column(avail[c].name, avail[c].sql_dtype) for c in extr_column_names]
            except KeyError as e:
                raise KeyError(f"Measurement group '{mg_name}' requested extraction columns {extr_column_names},\n"
                               f"but only the following are available: {list(avail.keys())},\nerrored on {str(e)}") from e
            extr_tab = Table(
                f'Extr -- {mg_name}', self.metadata,
                Column('loadid',INTEGER,nullable=False),
                Column('measid',INTEGER,nullable=False),
                ForeignKeyConstraint(columns=['loadid','measid'],**_CASC,
                                     refcolumns=[meas_tab.c.loadid, meas_tab.c.measid]),
                PrimaryKeyConstraint('loadid','measid'),
                *extr_columns, schema=self.int_schema)
        
        sweep_tab = self.metadata.tables.get(self.int_schema+f'.Sweep -- {mg_name}')
        if sweep_tab is None: sweep_tab = \
            Table(
                f'Sweep -- {mg_name}', self.metadata,
                Column('loadid', INTEGER, nullable=False),
                Column('measid', INTEGER, nullable=False),
                Column('sweep', BYTEA, nullable=False),
                Column('header', VARCHAR, nullable=False),
                Column('israw', BOOLEAN, nullable=False),
                PrimaryKeyConstraint('loadid', 'measid', 'header'),
                ForeignKeyConstraint(columns=['loadid', 'measid'], **_CASC,
                                     refcolumns=[meas_tab.c.loadid, meas_tab.c.measid]),
                schema=self.int_schema)

        return {'meas': meas_tab, 'extr': extr_tab, 'sweep': sweep_tab}
    
    def get_higher_analysis_dbtables(self,an_name: str) -> dict[str,Table]:
        from datavac.config.data_definition import DDEF
        an=DDEF().higher_analyses[an_name]
        aidt=self.metadata.tables.get(DBSTRUCT().int_schema+f'.AnalysisID -- {an_name}')
        ALL_MGS = DDEF().measurement_groups
        ALL_HAS = DDEF().higher_analyses
        if aidt is None:
            sampletab=DBSTRUCT().get_sample_dbtable()
            reqlids=[Column(f'loadid - {mg_name}',INTEGER,
                            ForeignKey(ALL_MGS[mg_name].trove().dbtables('loads').c.loadid,**_CASC),nullable=False,index=True)
                                for mg_name in an.required_dependencies if mg_name in ALL_MGS]
            reqaids=[Column(f'anlsid - {an2_name}',INTEGER,
                            ForeignKey(ALL_HAS[an2_name].dbtables('aidt').c.anlsid,**_CASC),nullable=False,index=True)
                                for an2_name in an.required_dependencies if an2_name in ALL_HAS]
            optlids=[Column(f'loadid - {mg_name}',INTEGER,
                            ForeignKey(ALL_MGS[mg_name].trove().dbtables('loads').c.loadid,**_CASC),nullable=True,index=True)
                                for mg_name in an.optional_dependencies if mg_name in ALL_MGS]
            optaids=[Column(f'anlsid - {an2_name}',INTEGER,
                            ForeignKey(ALL_HAS[an2_name].dbtables('aidt').c.anlsid,**_CASC),nullable=True,index=True)
                                for an2_name in an.optional_dependencies if an2_name in ALL_HAS]
            aidt=Table(f"AnalysisID -- {an_name}", self.metadata,
                       Column('anlsid', INTEGER, primary_key=True, autoincrement=True),
                       Column('sampleid', INTEGER, ForeignKey(sampletab.c['sampleid'], **_CASC), nullable=False),
                       *reqlids, *reqaids, *optlids, *optaids,
                       schema=self.int_schema)
        
        anls=self.metadata.tables.get(DBSTRUCT().int_schema+f'.Analysis -- {an_name}')
        if anls is None:
            try:
                subsample_references={ssr_name: self.datadef.subsample_references[ssr_name]
                                        for ssr_name in an.subsample_reference_names}
            except KeyError as e:
                raise KeyError(f"Analysis '{an_name}' requested subsample references {an.subsample_reference_names}, "
                               f"but only the following are available: {list(self.datadef.subsample_references.keys())}, "
                               f"errored on {str(e)}") from e
            subsample_reference_columns = \
                [Column(ssr.key_column.name, ssr.key_column.sql_dtype,
                             ForeignKey(self.get_subsample_reference_dbtable(ssr_name).c[ssr.key_column.name],
                                        name=f'fk_{ssr.key_column.name} -- {an_name}',**_CASC),nullable=False)
                         for ssr_name, ssr in subsample_references.items()]
            anls_column_names = an.analysis_column_names
            avail=an.available_analysis_columns()
            try:
                anls_columns = [Column(avail[c].name, avail[c].sql_dtype) for c in anls_column_names]
            except KeyError as e:
                raise KeyError(f"Analysis '{an_name}' requested columns {anls_column_names},\n"
                               f"but only the following are available: {list(avail.keys())},\nerrored on {str(e)}") from e
            anls=Table(f'Analysis -- {an_name}', self.metadata,
                Column('anlsid', INTEGER, ForeignKey(aidt.c.anlsid, **_CASC), nullable=False),
                Column('anlssubid', INTEGER, nullable=False),
                *subsample_reference_columns,
                *anls_columns,
                PrimaryKeyConstraint('anlsid', 'anlssubid'),
                schema=self.int_schema)
        return {'aidt': aidt, 'anls': anls}

    def get_higher_analysis_reload_table(self) -> Table:
        """Returns the reload table for higher analyses."""
        t=self.metadata.tables.get(self.int_schema+f'.ReAnls')
        return t if (t is not None) else \
            Table('ReAnls', self.metadata,
                  Column('sampleid', INTEGER, ForeignKey(self.get_sample_dbtable().c.sampleid, **_CASC), nullable=False),
                  Column('Analysis', VARCHAR, nullable=False),
                  UniqueConstraint('sampleid', 'Analysis'),
                  schema=self.int_schema)

    def get_blob_store_dbtable(self) -> Table:
        t=self.metadata.tables.get(self.int_schema+f'.Blob Store')
        return t if (t is not None) else \
            Table('Blob Store', self.metadata,
                    Column('name', VARCHAR, primary_key=True),
                    Column('blob', BYTEA, nullable=False),
                    Column('date_stored', TIMESTAMP, nullable=False),
                    schema=self.int_schema)