from __future__ import annotations
import io
from typing import TYPE_CHECKING, Optional
from datavac.util.dvlogging import time_it
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy import Connection


def upload_csv(df: pd.DataFrame, conn: Connection, schema: Optional[str], table: str):

    from datavac.database.db_structure import DBSTRUCT
    if schema:
        assert [c for c in df.columns]==[c.name for c in DBSTRUCT().metadata.tables[f'{schema}.{table}'].columns],\
            f"Column names in DataFrame do not match table {schema}.{table} in database. "\
            f"DataFrame columns:\n{list(df.columns)}\nTable columns:\n{[c.name for c in DBSTRUCT().metadata.tables[f'{schema}.{table}'].columns]}"

    with time_it(f"Conversion to csv for {table}",.1):
        output = io.StringIO()
        df.replace([np.inf,-np.inf],np.nan).to_csv(output, sep='|', header=False, index=False)
        output.seek(0)
    
    with time_it(f"Upload of csv for {table}",.1):
        with conn.connection.cursor() as cur: # type: ignore
            try:
                cur.copy_expert(f'COPY {schema+"." if schema else ""}"{table}" FROM STDIN WITH DELIMITER \'|\' NULL \'\'',file=output)
            except Exception as e:
                import pdb; pdb.set_trace()
                raise

def upload_binary(df:pd.DataFrame, conn: Connection, schema: str, table: str, override_converters: dict={}):
    from datavac.database.postgresql_binary_format import df_to_pgbin
    with time_it(f"Conversion to binary for {table}",.1):
        bio=df_to_pgbin(df, override_converters=override_converters)
        #print("BIO len:",len(bio.read()))
        #bio.seek(0)
        #print(bio.read().hex())
        #bio.seek(0)

    with time_it(f"Upload of binary for {table}",.1):
        with conn.connection.cursor() as cur: # type: ignore
            cur.copy_expert(f'COPY {schema}."{table}" FROM STDIN BINARY',bio)
    ##### TEMPORARILY REMOVING DB-API COMMIT
    #conn.connection.commit()