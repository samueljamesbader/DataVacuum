from typing import Optional
from datavac.database.db_upload_meas import enter_sample
import pandas as pd
from datavac.database.db_util import namewsq
from datavac.util.dvlogging import logger
from datavac.util.util import returner_context
from sqlalchemy import Connection, delete, select


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
    from datavac.config.project_config import PCONF
    from datavac.database.db_connect import get_engine_rw
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