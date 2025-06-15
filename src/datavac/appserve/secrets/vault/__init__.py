from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datavac.database.db_connect import PostgreSQLConnectionInfo


class Vault():
    def get_db_connection_info(self, usermode: str = 'ro') -> 'PostgreSQLConnectionInfo':
        """Fetches the connection information for the database from the vault.
        Args:
            usermode (str): The user mode, one of ['ro', 'rw', 'so']. Defaults to 'ro'.
        """
        assert usermode in ['ro', 'rw', 'so']
        raise NotImplementedError("Subclass should implement")