from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.engine import URL
    from sqlalchemy import Engine

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
    
def get_engine_ro():
    from datavac.config.project_config import PCONF
    return _make_engine(PCONF().vault.get_db_connection_info('ro'))
    
def get_engine_so():
    from datavac.config.project_config import PCONF
    return _make_engine(PCONF().vault.get_db_connection_info('so'))
    

def _make_engine(connection_info: PostgreSQLConnectionInfo) -> 'Engine':
    from sqlalchemy import create_engine
    url=URL.create(
        drivername=connection_info.driver,
        username=connection_info.username,
        password=connection_info.password,
        host=connection_info.host,
        port=connection_info.port,
        database=connection_info.database,
    )
    return create_engine(url, connect_args=connection_info.sslargs, pool_recycle=60)