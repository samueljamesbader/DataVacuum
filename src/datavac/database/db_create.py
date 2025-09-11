from typing import Optional
from datavac.config.data_definition import DDEF
from datavac.config.project_config import PCONF
from datavac.database.db_connect import DBConnectionMode, avoid_db_if_possible, get_db_connection_info,\
      get_engine_so, raw_psycopg2_connection_do, raw_psycopg2_connection_so, have_do_creds
from datavac.database.db_structure import DBSTRUCT
from datavac.util.dvlogging import logger
from sqlalchemy import Connection, text

def ensure_database_existence():
    """Ensure the database exists.

    If database owner credentials are available and the database does not exist, it will be created.

    Returns:
        bool: True if the database pre-existed, False if it was freshly created.
    """
    # If we have database owner credentials, use them to ensure database existence
    # Otherwise, we can only assume it exists (and will error out if it fails)
    from datavac.config.project_config import PCONF
    if have_do_creds():
        with raw_psycopg2_connection_do(override_db='postgres') as con:
            con.autocommit = True

            with con.cursor() as cur:

                # Check if the database exists
                dbname=get_db_connection_info(DBConnectionMode.DATABASE_OWNER).database
                cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
                pre_exists = cur.fetchone()

                # If not, create the database
                if not pre_exists:
                    from psycopg2 import sql
                    logger.debug("Database does not exist, creating it.")
                    cur.execute(sql.SQL('CREATE DATABASE {};').format(sql.Identifier(dbname)))
    else:
        with raw_psycopg2_connection_so() as con:
            try:
                with con.cursor() as cur: cur.execute(f"SELECT 1")
                pre_exists = True
            except Exception as e:
                logger.critical("Failure to ensure database existence.")
                raise
    return pre_exists

def ensure_clear_database():
    """Ensure the database exists and is clear."""
    pre_exists = ensure_database_existence()

    # If it pre-existed, clear it
    if pre_exists:
        with raw_psycopg2_connection_so() as con:
            con.autocommit = True
            with con.cursor() as cur:
                logger.debug("Database exists, clearing it.")

                # Drop all user-defined schemas 
                cur.execute("""
                DO $$
                DECLARE
                    schema_name_var text;
                BEGIN
                    FOR schema_name_var IN
                        SELECT schema_name FROM information_schema.schemata
                        WHERE schema_name NOT IN ('public', 'pg_catalog', 'information_schema','pg_toast')
                    LOOP
                        EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE', schema_name_var);
                    END LOOP;
                END $$;
                """)


def _setup_foundation(conn:Connection):
    """Set up the foundation of the database, including schemas and the blob store table."""

    # Get the usernames for the different roles
    ro_user=get_db_connection_info(DBConnectionMode.READ_ONLY).username
    rw_user=get_db_connection_info(DBConnectionMode.READ_WRITE).username
    so_user=get_db_connection_info(DBConnectionMode.SCHEMA_OWNER).username

    # Make fresh and ensure good default permissions
    make_schemas=" ".join([f"CREATE SCHEMA IF NOT EXISTS {schema}; " \
                           f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO PUBLIC; " \
                           f"GRANT USAGE ON SCHEMA {schema} TO PUBLIC; " \
                           f"GRANT TEMP ON DATABASE {get_db_connection_info(DBConnectionMode.READ_WRITE).database} TO {rw_user}; " \
                           f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {rw_user}; "\
                           f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO {rw_user}; "\
                           f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT ALL PRIVILEGES ON TABLES TO {so_user}; "\
                           f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT ALL PRIVILEGES ON SEQUENCES TO {so_user}; "\
                           f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT ON TABLES TO {ro_user}; "
                           for schema in [DBSTRUCT().int_schema,DBSTRUCT().jmp_schema]])
    
    # Handle privleges on any pre-existing tables
    update_schemas=" ".join([f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {rw_user}; "\
                             f"GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA {schema} TO {rw_user}; "\
                             f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema} TO {so_user}; "\
                             f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {schema} TO {so_user}; "\
                             f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {ro_user}; "
                             for schema in [DBSTRUCT().int_schema,DBSTRUCT().jmp_schema]])

    # Set the search path to the internal schema               
    set_search_path = f"SET SEARCH_PATH={DBSTRUCT().int_schema};"
    from sqlalchemy.schema import CreateTable

    # Ensure blob store table exists
    create_blob_store = str(CreateTable(DBSTRUCT().get_blob_store_dbtable()).compile(conn))\
        .replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")

    # Execute all the setup commands
    conn.execute(text(make_schemas+update_schemas+set_search_path+create_blob_store))

def create_meas_group_view(mg_name: str, conn: Optional[Connection]=None, just_DDL_string: bool = False) -> str:
    from datavac.database.db_get import get_table_depends_and_hints_for_meas_group, joined_select_from_dependencies
    mg=DDEF().measurement_groups[mg_name]
    meas_tab=DBSTRUCT().get_measurement_group_dbtables(mg_name)['meas']
    td, jh = get_table_depends_and_hints_for_meas_group(mg, include_sweeps=False)
    sel,_=joined_select_from_dependencies(columns=None, absolute_needs=[meas_tab],
                                        table_depends=td, pre_filters={},join_hints=jh)
    seltextl=sel.compile(conn, compile_kwargs={"literal_binds": True})
    view_namewsq = f'{DBSTRUCT().jmp_schema}."{mg_name}"'
    ddl=f"""CREATE OR REPLACE VIEW {view_namewsq} AS {seltextl}"""
    if not just_DDL_string:
        assert conn is not None, "Connection must be provided if not just generating DDL string."
        # sel.compile converts % in column names to %%
        # https://github.com/sqlalchemy/sqlalchemy/discussions/8077
        conn.execute(text(ddl.replace("%%","%")))
    return ddl

def create_analysis_view(an_name: str, conn: Optional[Connection]=None, just_DDL_string: bool = False) -> str:
    from datavac.database.db_get import get_table_depends_and_hints_for_analysis, joined_select_from_dependencies
    an=DDEF().higher_analyses[an_name]
    anls_tab=DBSTRUCT().get_higher_analysis_dbtables(an.name)['anls']
    td, jh = get_table_depends_and_hints_for_analysis(an)
    sel,_=joined_select_from_dependencies(columns=None, absolute_needs=[anls_tab],
                                        table_depends=td, pre_filters={},join_hints=jh)
    seltextl=sel.compile(conn, compile_kwargs={"literal_binds": True})
    view_namewsq = f'{DBSTRUCT().jmp_schema}."{an_name}"'
    ddl=f"""CREATE OR REPLACE VIEW {view_namewsq} AS {seltextl}"""
    if not just_DDL_string:
        assert conn is not None, "Connection must be provided if not just generating DDL string."
        # sel.compile converts % in column names to %%
        # https://github.com/sqlalchemy/sqlalchemy/discussions/8077
        conn.execute(text(ddl.replace("%%","%"))) 
    return ddl


def create_all():
    """Create all database tables and schemas."""
    with avoid_db_if_possible():
        with get_engine_so().begin() as conn:
            _setup_foundation(conn)

        for sr_name in PCONF().data_definition.sample_references:
            DBSTRUCT().get_sample_reference_dbtable(sr_name)
            
        DBSTRUCT().get_sample_dbtable()

        for sd_name in PCONF().data_definition.sample_descriptors:
            DBSTRUCT().get_sample_descriptor_dbtable(sd_name)

        for ssr_name in PCONF().data_definition.subsample_references:
            DBSTRUCT().get_subsample_reference_dbtable(ssr_name)

        for mg_name in PCONF().data_definition.measurement_groups:
            DBSTRUCT().get_measurement_group_dbtables(mg_name)
            
        for an_name in PCONF().data_definition.higher_analyses:
            DBSTRUCT().get_higher_analysis_dbtables(an_name)

        DBSTRUCT().get_higher_analysis_reload_table()

        DBSTRUCT().metadata.create_all(get_engine_so(), checkfirst=True)

        with get_engine_so().begin() as conn:
            for mg_name in PCONF().data_definition.measurement_groups:
                 create_meas_group_view(mg_name, conn)
            for an_name in PCONF().data_definition.higher_analyses:
                create_analysis_view(an_name, conn)
        
        DDEF().populate_initial()