from typing import TYPE_CHECKING, Callable
import argparse

from datavac.database.db_connect import DBConnectionMode

if TYPE_CHECKING:
    from datavac.database.db_connect import PostgreSQLConnectionInfo


class Vault():
    def get_db_connection_info(self, usermode: DBConnectionMode = DBConnectionMode.READ_ONLY) -> 'PostgreSQLConnectionInfo':
        """Fetches the connection information for the database from the vault."""
        raise NotImplementedError("Subclass should implement")
    
    def clear_vault_cache(self):
        """Clears any potential local vault cache."""
        pass

    def _cli_clear_vault_cache(self,*args):
        parser=argparse.ArgumentParser(description='Clear local vault cache')
        namespace=parser.parse_args(args)
        self.clear_vault_cache()
        print("Cleared vault cache")

    def get_cli_funcs(self) -> dict[str, Callable]:
        """Returns a dictionary of CLI functions for CyberArk vault operations."""
        return {
            'clear_cache': self._cli_clear_vault_cache,
        }

from datavac.util.cli import CLIIndex
def _get_CLI_funcs():
    from datavac.config.project_config import PCONF
    return PCONF().vault.get_cli_funcs()
VAULT_CLI = CLIIndex(cli_funcs=_get_CLI_funcs)