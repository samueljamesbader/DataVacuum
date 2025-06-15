import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from datavac.appserve.secrets.user_side import get_secret_from_deployment
from datavac.config.project_config import PCONF, is_server
from datavac.database.db_connect import PostgreSQLConnectionInfo
from datavac.util.logging import logger

@cache
def get_db_connection_info(usermode: str = 'ro') -> 'PostgreSQLConnectionInfo':
    """Fetches the connection information for the database.
    Args:
        usermode (str): The user mode, one of ['ro', 'rw', 'so']. Defaults to 'ro'.
    """
    assert usermode in ['ro', 'rw', 'so']
    last_error: Optional[Exception] = None

    try: return _get_db_connection_info_from_environment(usermode)
    except KeyError as e: last_error = e

    try: return PCONF().vault.get_db_connection_info(usermode)
    except NotImplementedError as e: pass
    except Exception as e: last_error = e
    
    if not is_server():
        try:
            conn_str=get_secret_from_deployment({'ro':'readonly_dbstring', 'rw': 'readwrite_dbstring', 'so': 'super_dbstring'}[usermode])
            return PostgreSQLConnectionInfo.from_connection_string(conn_str)
        except Exception as e: last_error = e

    logger.error(f"Failed to fetch database connection info for usermode '{usermode}': {last_error}")
    raise last_error

def _get_db_connection_info_from_environment(usermode: str = 'ro') -> 'PostgreSQLConnectionInfo':
    """Fetches the connection information for the database from the environment.
    Args:
        usermode (str): The user mode, one of ['ro', 'rw', 'so']. Defaults to 'ro'.
    """
    assert usermode in ['ro', 'rw', 'so']

    if f'DATAVACUUM_DB_{usermode.upper()}_CONNECTION_STRING' not in os.environ:
        raise KeyError(
            f"Default SecretsFetcher.get_ro_connection_info(usermode={usermode}) requires "\
                f"the environment variable DATAVACUUM_DB_{usermode.upper()}_CONNECTION_STRING."
        )
    return PostgreSQLConnectionInfo.from_connection_string(
                os.environ[f'DATAVACUUM_DB_{usermode}_CONNECTION_STRING'])