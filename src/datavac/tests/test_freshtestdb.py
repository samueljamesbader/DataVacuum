import os
import time
import psycopg2
import pytest
from psycopg2 import sql
from datavac.util.caching import cli_clear_cache
from datavac.util.logging import logger


@pytest.fixture(scope='session',autouse=True)
def make_fresh_testdb():
    logger.debug("Creating test database")
    con = psycopg2.connect(dbname='postgres', user='postgres', host='', password=os.environ['DATAVACUUM_TEST_DB_PASS'])
    con.autocommit = True

    cur = con.cursor()

    # Check if the database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'datavacuum_test'")
    exists = cur.fetchone()

    if exists:
        # Drop all schemas except 'pg_catalog' and 'information_schema'
        cur.execute("""
        DO $$
        DECLARE
            schema_name_var text;
        BEGIN
            FOR schema_name_var IN
                SELECT schema_name FROM information_schema.schemata
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema','pg_toast')
            LOOP
                EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE', schema_name_var);
            END LOOP;
        END $$;
        """)
    else:
        # Create the database
        cur.execute(sql.SQL('CREATE DATABASE {};').format(sql.Identifier('datavacuum_test')))

    cur.close()
    con.close()

    logger.debug("Created test database.  Populating skeleton tables")

    from datavac.io.database import get_database
    db = get_database(metadata_source='reflect')
    db.validate(on_mismatch='replace')
    logger.debug("Fresh test database ready for data.")

@pytest.fixture(scope='session',autouse=True)
def clear_cache():
    cli_clear_cache()
