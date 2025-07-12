from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any, Generator, Optional

from datavac.util.dvlogging import logger

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
            part.split('=')[0].strip().lower()\
                        .replace("uid","username").replace("server","host")
                    :part.split('=')[1].strip()
                for part in conn_str.split(';') if '=' in part}
        return PostgreSQLConnectionInfo(**parts)
    
    def to_connection_string(self) -> str:
        """Returns a connection string representation of the connection info."""
        parts = [
            f"username={self.username}",
            f"password={self.password}",
            f"host={self.host}",
            f"port={self.port}",
            f"database={self.database}"
        ]
        return ';'.join(parts)
    
    def __str__(self) -> str:
        """Returns a string representation of the connection info."""
        return f"[{self.username}@{self.host}:{self.port} for '{self.database}']"

@cache 
def get_engine_ro():
    return _make_engine(get_db_connection_info(DBConnectionMode.READ_ONLY))

@cache 
def get_engine_rw():
    return _make_engine(get_db_connection_info(DBConnectionMode.READ_WRITE))

@cache 
def get_engine_so():
    return _make_engine(get_db_connection_info(DBConnectionMode.SCHEMA_OWNER))

@contextmanager
def raw_psycopg2_connection_so() -> Generator['psycopg2.extensions.connection',None, None]:
    connection_info = get_db_connection_info(DBConnectionMode.SCHEMA_OWNER)
    with _raw_psycopg2_connection(connection_info) as conn: yield conn

@contextmanager
def raw_psycopg2_connection_do(override_db:Optional[str]=None) -> Generator['psycopg2.extensions.connection',None, None]:
    connection_info = get_db_connection_info(DBConnectionMode.DATABASE_OWNER)
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

def get_db_connection_info(min_usermode: DBConnectionMode = DBConnectionMode.READ_ONLY) -> 'PostgreSQLConnectionInfo':
    """Fetches the connection information for the database, ensuring it is at least the specified usermode.
    
    Args:
        min_usermode: The minimum user mode required for the connection.
    
    """
    at_least=False
    for usermode in DBConnectionMode: # iterate in order of usermode
        if usermode == min_usermode: at_least=True
        if at_least:
            try: 
                ci = get_specific_db_connection_info(usermode)
                if usermode != min_usermode:
                    logger.warning(f"Couldn't find connection info for {min_usermode.name}, using {usermode.name} instead.")
                return ci
            except Exception as e:
                logger.warning(f"Failed to get connection info for {usermode.name}: {e}, escalating...")
    raise PermissionError(
        f"Failed to get database connection info for any usermode >= {min_usermode.name}. "
        "Check your environment variables or project configuration."
    )

@cache
def get_specific_db_connection_info(usermode: DBConnectionMode = DBConnectionMode.READ_ONLY) -> 'PostgreSQLConnectionInfo':
    """Fetches the connection information for the database.
    
    The search order is as follows:
    1. Environment variable `DATAVACUUM_DB_{usermode}_CONNECTION_STRING` if set
    2. Environment variable `DATAVACUUM_DB_CONNECTION_STRING` if set
    3. Project configuration vault if configured
    (If the context name contains 'local', the search stops here, otherwise...)
    4. Asking the deployment server (if this code is not already in server mode)

    Args:
        usermode: which set of credentials to use
    """
    last_error: Optional[Exception] = None

    try: return _get_db_connection_info_from_environment(usermode)
    except KeyError as e: last_error = e

    from datavac.config.project_config import PCONF, is_server
    from datavac.config.contexts import get_current_context_name
    
    try: return PCONF().vault.get_db_connection_info(usermode)
    except NotImplementedError as e: pass
    except Exception as e: last_error = e
    
    if ('local' in (get_current_context_name() or '')): raise last_error
    if ('builtin:' in (get_current_context_name() or '')): raise last_error
    if not is_server():
        from datavac.appserve.dvsecrets.user_side import get_secret_from_deployment
        try:
            conn_str=get_secret_from_deployment({DBConnectionMode.READ_ONLY:'readonly_dbstring',
                                                 DBConnectionMode.READ_WRITE: 'readwrite_dbstring',
                                                 DBConnectionMode.SCHEMA_OWNER: 'super_dbstring'}[usermode])
            return PostgreSQLConnectionInfo.from_connection_string(conn_str)
        except Exception as e: last_error = e

    logger.error(f"Failed to fetch database connection info for usermode '{usermode}': {last_error}")
    raise last_error

def _get_db_connection_info_from_environment(usermode: DBConnectionMode) -> 'PostgreSQLConnectionInfo':
    """Fetches the connection information for the database from the environment.
    Args:
        usermode (str): The user mode, one of ['ro', 'rw', 'so']. Defaults to 'ro'.
    """
    import os
    envvar1=f'DATAVACUUM_DB_{usermode.value.upper()}_CONNECTION_STRING'    
    envvar2=f'DATAVACUUM_DB_CONNECTION_STRING'
    conn_str = os.environ.get(envvar1) or os.environ.get(envvar2)
    if conn_str is None:
        raise KeyError(f"Environment variables {envvar1} AND/OR {envvar2} not found.")
    return PostgreSQLConnectionInfo.from_connection_string(conn_str)


def have_do_creds() -> bool:
    """Checks if we have credentials for the database owner (do) user."""
    try:
        connection_info=get_db_connection_info(DBConnectionMode.DATABASE_OWNER)
        return connection_info is not None
    except KeyError: return False
    except NotImplementedError: return False
