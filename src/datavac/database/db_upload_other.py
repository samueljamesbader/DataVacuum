from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from datavac.config.data_definition import HigherAnalysis
from datavac.database.db_upload_meas import enter_sample
from datavac.measurements.measurement_group import MeasurementGroup
import pandas as pd
from datavac.database.db_util import namewsq
from datavac.util.dvlogging import logger
from datavac.util.util import returner_context

if TYPE_CHECKING:
    from sqlalchemy import Connection


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
    from sqlalchemy import text, delete
    from sqlalchemy.schema import CreateTable, DropTable
    from datavac.database.db_connect import get_engine_rw, get_engine_so
    from datavac.database.db_create import create_meas_group_view, create_analysis_view
    ssr = PCONF().data_definition.subsample_references[ssr_name]
    ssrtab = DBSTRUCT().get_subsample_reference_dbtable(ssr_name)
    with (returner_context(conn) if conn else get_engine_rw().begin()) as conn: 

        # Load the data into a temporary table and check if it is different
        create_temp_table="CREATE TEMP TABLE tmplay ("+str(CreateTable(ssrtab).compile(conn)).split("\" (",maxsplit=1)[1]+";"
        table_cols_same=(conn.execute(text(create_temp_table+\
            "WITH A AS (SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'tmplay'), "\
            "B AS (SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = :tblname) "\
            "SELECT CASE WHEN EXISTS (SELECT * FROM A EXCEPT SELECT * FROM B) OR EXISTS (SELECT * FROM B EXCEPT SELECT * FROM A) "\
            "THEN 'different' ELSE 'same' END AS result ;").bindparams(tblname=ssr_name)).all()[0][0] == 'same')
        if not table_cols_same: logger.debug(f"Column structure changed for {ssr_name}, updating")
        upload_csv(data, conn, None, 'tmplay')
        if table_cols_same and conn.execute(text(
            f'''SELECT CASE WHEN EXISTS (TABLE {namewsq(ssrtab)} EXCEPT TABLE tmplay)
              OR EXISTS (TABLE tmplay EXCEPT TABLE {namewsq(ssrtab)})
            THEN 'different' ELSE 'same' END AS result ;''')).all()[0][0] == 'same':
            logger.debug(f"Content unchanged for {ssr_name}")
            conn.execute(text(f'DROP TABLE tmplay;'))

        # If the content has changed, we need to update the table
        else:
            logger.debug(f"Content changed for {ssr_name}, updating")

            # Statements to temporarily remove foreign key constraints
            removal_statements: list[str] = []
            readdit_statements: list[str] = []

            # Statements to recreate views
            mg_views_to_recreate: list[str] = []
            an_views_to_recreate: list[str] = []

            # Populate the above statements lists
            for mgoa in list(DDEF().measurement_groups.values())+list(DDEF().higher_analyses.values()):
                if ssr_name in mgoa.subsample_reference_names:
                    if isinstance(mgoa, MeasurementGroup):
                        coretab=DBSTRUCT().get_measurement_group_dbtables(mgoa.name)['meas']
                        mg_views_to_recreate.append(create_meas_group_view(mgoa.name, just_DDL_string=True))
                    else:
                        coretab=DBSTRUCT().get_higher_analysis_dbtables(mgoa.name)['anls']
                        an_views_to_recreate.append(create_analysis_view(mgoa.name, just_DDL_string=True))
                    if dump_extractions_and_analyses:
                        raise NotImplementedError()
                    removal_statements.append(f'ALTER TABLE {namewsq(coretab)}'\
                                      f' DROP CONSTRAINT IF EXISTS "fk_{ssr.key_column.name} -- {mgoa.name}";')
                    readdit_statements.append(f'ALTER TABLE {namewsq(coretab)}' \
                                      f' ADD CONSTRAINT "fk_{ssr.key_column.name} -- {mgoa.name}" FOREIGN KEY ("{ssr.key_column.name}")' \
                                      f' REFERENCES {namewsq(ssrtab)} ("{ssr.key_column.name}") ON DELETE CASCADE;')
                    
            all_statements=[]

            # Remove foreign key constraints
            all_statements+=removal_statements

            # Bring in the new table
            if table_cols_same:
                all_statements.append(str(delete(ssrtab).compile(conn)))
            else:
                all_statements.append(str(DropTable(ssrtab).compile(conn))+" CASCADE")
                all_statements.append(str(CreateTable(ssrtab).compile(conn)))
            all_statements.append(f'INSERT INTO {namewsq(ssrtab)} SELECT * from tmplay;')

            # Recreate foreign key constraints
            all_statements+=readdit_statements
            all_statements+=mg_views_to_recreate
            all_statements+=an_views_to_recreate

            # Execute all statements in a single transaction
            all_statements.append(f'DROP TABLE tmplay;')
            conn.execute(text(";".join(all_statements)))


def upload_sample_descriptor(sd_name:str, data: pd.DataFrame, conn: Optional[Connection] = None, clear_all_previous: bool = False):
    """ Upload a sample descriptor to the database.

    Args:
        sd_name: The name of the sample descriptor.
        data: A DataFrame containing the sample descriptor data.
    """
    # TODO: Inefficient, should be done in O(1) queries
    import pandas as pd
    from datavac.database.db_structure import DBSTRUCT
    from datavac.database.postgresql_upload_utils import upload_csv
    from datavac.config.project_config import PCONF
    from datavac.config.data_definition import DDEF
    from datavac.database.db_connect import get_engine_rw
    from sqlalchemy import select, delete
    sampletab=DBSTRUCT().get_sample_dbtable()
    sdtab=DBSTRUCT().get_sample_descriptor_dbtable(sd_name)
    samplename_col=PCONF().data_definition.sample_identifier_column.name
    with (returner_context(conn) if conn else get_engine_rw().begin()) as conn:
        res=conn.execute(select(sampletab.c.sampleid,sampletab.c[samplename_col])\
                            .where(sampletab.c[samplename_col].in_(list(data.index)))).all()
        for samplename, row in data.iterrows():
            if samplename not in [r[1] for r in res]:
                # TODO: make a "bulk" enter_samples function that only requires one execution
                enter_sample(conn, **PCONF().data_definition.sample_info_completer(
                    dict(**{samplename_col:samplename},**{k:v for k,v in row.items() if k in sampletab.c}))) # type: ignore
        res=conn.execute(select(sampletab.c.sampleid,sampletab.c[samplename_col])\
                            .where(sampletab.c[samplename_col].in_(list(data.index)))).all()
        data=pd.merge(left=data,right=pd.DataFrame(res,columns=['sampleid',samplename_col]),
                 how='left',left_index=True,right_on=samplename_col).drop(columns=[samplename_col]).set_index('sampleid')
        #print(data)
        dlt_state=delete(sdtab)
        if not clear_all_previous: dlt_state=dlt_state.where(sdtab.c.sampleid.in_(list(data.index)))
        conn.execute(dlt_state)
        upload_csv(data.reset_index()[[c.name for c in sdtab.c]], conn, DBSTRUCT().int_schema, sd_name,)