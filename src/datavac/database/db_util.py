from __future__ import annotations
from typing import TYPE_CHECKING, Any

from datavac.database.db_connect import get_engine_ro
from datavac.util.util import returner_context
from datavac.appserve.api import client_server_split

if TYPE_CHECKING:
    import pandas as pd
    from pandas import DataFrame
    from sqlalchemy import Connection, Table
else:
    # Otherwise Pydantic's validate_call won't be able to parse the types
    Connection=Any
    DataFrame=Any

def run_query(query,conn=None, commit=False):
    from sqlalchemy import text 
    with (returner_context(conn) if conn else get_engine_ro().connect()) as conn:
        result=conn.execute(text(query)).all()
        if commit: conn.commit()
    return result

def read_sql(query,conn=None) -> DataFrame:
    import pandas as pd
    with (returner_context(conn) if conn else get_engine_ro().connect()) as conn:
        result=pd.read_sql(query,conn)
    return result

@client_server_split("read_only_sql", return_type="pd", split_on="direct_db_access")
def read_only_sql(query,conn=None) -> DataFrame:
    import pandas as pd
    with (returner_context(conn) if conn else get_engine_ro(only=True).connect()) as conn:
        result=pd.read_sql(query,conn)
    return result

def namews(table:Table) -> str:
    assert table.schema is not None, "Table schema must be set to use namews"
    return f"{table.schema}.{table.name}" if table.schema else table.name
def namewsq(table:Table) -> str:
    assert table.schema is not None, "Table schema must be set to use namewsq"
    return f"{table.schema}.\"{table.name}\"" if table.schema else table.name