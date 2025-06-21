from typing import TYPE_CHECKING

from datavac.database.db_connect import DBConnectionMode

if TYPE_CHECKING:
    from datavac.database.db_connect import PostgreSQLConnectionInfo


class Vault():
    def get_db_connection_info(self, usermode: DBConnectionMode = DBConnectionMode.READ_ONLY) -> 'PostgreSQLConnectionInfo':
        """Fetches the connection information for the database from the vault."""
        raise NotImplementedError("Subclass should implement")
    
    def have_do_creds(self) -> bool:
        """Checks if the vault has credentials for the database owner (do) user."""
        try:
            connection_info = self.get_db_connection_info(DBConnectionMode.DATABASE_OWNER)
            return connection_info is not None
        except KeyError: return False
        except NotImplementedError: return False
