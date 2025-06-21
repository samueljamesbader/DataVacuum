from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any, Generator, Optional

if TYPE_CHECKING:
    from sqlalchemy import Engine
    import psycopg2

class DBConnectionMode(Enum):
    """Enum to represent the different modes of database connection."""
    READ_ONLY = 'ro'
    READ_WRITE = 'rw'
    SCHEMA_OWNER = 'so'
    DATABASE_OWNER = 'do'


@dataclass
class PostgreSQLConnectionInfo():
    """Dataclass to hold the connection information for a PostgreSQL database."""
    username: str
    password: str
    host: str
    port: int
    database: str
    
    @property
    def driver(self)->str:
        return 'postgresql'
    
    @property
    def sslargs(self) -> dict:
        from datavac.config.project_config import PCONF
        rootcert_path = PCONF().cert_depo.get_ssl_rootcert_path_for_db()
        if rootcert_path:
            return {'sslmode': 'verify-full', 'sslrootcert': rootcert_path}
        else:
            return {}
    
    @staticmethod
    def from_connection_string(conn_str: str, sslargs: dict[str, str] = {}) -> 'PostgreSQLConnectionInfo':
        """Creates a PostgreSQLConnectionInfo from a connection string."""
        parts: dict[str,Any] = {
            part.split('=')[0].lower().replace("uid","username").replace("server","host"):part.split('=')[1]
                 for part in conn_str.split(';') if '=' in part}
        parts['port'] = int(parts['port'])
        return PostgreSQLConnectionInfo(**parts)
    
    def __str__(self) -> str:
        """Returns a string representation of the connection info."""
        return f"[{self.username}@{self.host}:{self.port} for '{self.database}']"

@cache 
def get_engine_ro():
    from datavac.config.project_config import PCONF
    return _make_engine(PCONF().vault.get_db_connection_info(DBConnectionMode.READ_ONLY))

@cache 
def get_engine_so():
    from datavac.config.project_config import PCONF
    return _make_engine(PCONF().vault.get_db_connection_info(DBConnectionMode.SCHEMA_OWNER))

@contextmanager
def raw_psycopg2_connection_so() -> Generator['psycopg2.extensions.connection',None, None]:
    from datavac.config.project_config import PCONF
    connection_info = PCONF().vault.get_db_connection_info(DBConnectionMode.SCHEMA_OWNER)
    with _raw_psycopg2_connection(connection_info) as conn: yield conn

@contextmanager
def raw_psycopg2_connection_do(override_db:Optional[str]=None) -> Generator['psycopg2.extensions.connection',None, None]:
    from datavac.config.project_config import PCONF
    connection_info = PCONF().vault.get_db_connection_info(DBConnectionMode.DATABASE_OWNER)
    if override_db is not None:
        connection_info = deepcopy(connection_info)
        connection_info.database = override_db
    with _raw_psycopg2_connection(connection_info) as conn: yield conn

def _make_engine(connection_info: PostgreSQLConnectionInfo) -> 'Engine':
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    url=URL.create(
        drivername=connection_info.driver,
        username=connection_info.username,
        password=connection_info.password,
        host=connection_info.host,
        port=connection_info.port,
        database=connection_info.database,
    )
    return create_engine(url, connect_args=connection_info.sslargs, pool_recycle=60)

@contextmanager
def _raw_psycopg2_connection(connection_info: PostgreSQLConnectionInfo) -> Generator['psycopg2.extensions.connection',None, None]:
    """Creates a raw psycopg2 connection using the provided connection info."""
    import psycopg2
    conn = psycopg2.connect(
            dbname=connection_info.database,
            user=connection_info.username,
            password=connection_info.password,
            host=connection_info.host,
            port=connection_info.port,
            **connection_info.sslargs
    )
    try: yield conn
    finally: conn.close()
