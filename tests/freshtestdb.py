import os
import time
import psycopg2
import pytest
from psycopg2 import sql
from datavac.util.logging import logger


def make_fresh_testdb(dbname):
    logger.debug("Creating test database")
    con = psycopg2.connect(dbname='postgres', user='postgres', host='localhost', password=os.environ['DATAVACUUM_TEST_DB_PASS'])
    con.autocommit = True

    cur = con.cursor()

    # Check if the database exists
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
    exists = cur.fetchone()

    if exists:

        # Connect to actual database
        cur.close()
        con.close()
        con = psycopg2.connect(dbname=dbname, user='postgres', host='localhost', password=os.environ['DATAVACUUM_TEST_DB_PASS'])
        con.autocommit = True
        cur = con.cursor()

        logger.debug("Database exists, clearing it.")
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
        cur.execute(sql.SQL('CREATE DATABASE {};').format(sql.Identifier(dbname)))

    cur.close()
    con.close()

    logger.debug("Created or cleared test database.  Populating skeleton tables")

    from datavac.io.database import get_database, unget_database
    unget_database()  # Clear any existing database connection
    db = get_database(metadata_source='reflect')
    db.validate(on_mismatch='replace')
    logger.debug(f"Fresh test database {dbname} ready for data.")
