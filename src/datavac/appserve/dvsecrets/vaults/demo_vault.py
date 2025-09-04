from datavac.appserve.dvsecrets.vaults.vault import Vault
from datavac.database.db_connect import DBConnectionMode, PostgreSQLConnectionInfo


import os
from dataclasses import dataclass


@dataclass
class DemoVault(Vault):
    """A vault implementation for the demo project that provides hardcoded connection info."""
    dbname: str

    def get_db_connection_info(self, usermode: DBConnectionMode = DBConnectionMode.READ_ONLY) -> 'PostgreSQLConnectionInfo':
        """Returns connection info for a local demo database for which user is assumed to have full owner permissions."""
        from datavac.database.db_connect import PostgreSQLConnectionInfo
        return PostgreSQLConnectionInfo(
            username='postgres',
            host='localhost',
            port=5432,
            database=self.dbname,
            password=os.environ.get('DATAVACUUM_TEST_DB_PASS', 'insecure_default_password')
        )
    def get_access_key_sign_seed(self) -> bytes:
        return b'Demo'
    def get_auth_info(self) -> dict[str,str]: 
        return {'oauth_provider':'none'}