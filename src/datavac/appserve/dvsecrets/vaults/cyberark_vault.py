from dataclasses import dataclass, field
from functools import cache
import os
from typing import TYPE_CHECKING, Callable, Optional, overload
from datavac.appserve.dvsecrets.vaults.vault import Vault
from datavac.config.project_config import PCONF
from datavac.database.db_connect import DBConnectionMode, PostgreSQLConnectionInfo
from datavac.util.caching import pickle_cached
from datavac.util.dvlogging import time_it

@dataclass
class CyberArkVault(Vault):

    api_url: str
    appid: str
    safe: str
    cache_on_disk: bool = False

    _getter: Callable = field(init=False, repr=False)
    def __post_init__(self):
        func = _raw_get_vault_response_text
        if self.cache_on_disk:
            func= pickle_cached(cache_dir="vault",
                                namer=lambda api_url,appid,safe,account_name:\
                                    f"{appid}_{safe}_{account_name}.pkl")(func)
        setattr(self,'_getter',cache(func))
    
    def clear_vault_cache(self):
        """Clears the vault cache."""
        from datavac.util.caching import clear_local_cache
        clear_local_cache("vault")

    def get_db_connection_info(self, usermode: DBConnectionMode = DBConnectionMode.READ_ONLY) -> 'PostgreSQLConnectionInfo':
        """Fetches the connection information for the database from the CyberArk vault.
        
        # TODO: This assumption should be removed or simplified
        Assumes the name of the secret in the vault is '{deployment_name}-read' or '{deployment_name}-super' depending on the usermode.
        """
        deployment_name=PCONF().deployment_name
        return PostgreSQLConnectionInfo(**dict(zip(
            ['username','password','host','port','database'],
            self.get_from_vault(f'{deployment_name}-{ {DBConnectionMode.READ_ONLY:"read",
                                                       DBConnectionMode.READ_WRITE:"rw",
                                                       DBConnectionMode.SCHEMA_OWNER:"super"}[usermode]}',
                                ['UserName','Content','Address','Port','Database'])))) # type: ignore
    
    def get_access_key_sign_seed(self) -> bytes:
        return self.get_from_vault('AccessKeySignSeed','Content').encode()
    
    def get_auth_info(self):
        # TODO: remove or generalize
        deployment_uri=PCONF().deployment_uri
        if ('localhost' in deployment_uri) and not (os.environ.get("DATAVACUUM_OAUTH_PROVIDER",None)=='azure'):
            return {'oauth_provider':'none',}
        else:
            return {
                'oauth_provider':'azure',
                'oauth_key':self.get_from_vault('DataVacuumSSO','ApplicationID'),
                'oauth_extra_params':{'tenant_id':self.get_from_vault('DataVacuumSSO','ActiveDirectoryID')},
                'oauth_secret':self.get_from_vault('DataVacuumSSO','Content'),
                'oauth_redirect_uri':deployment_uri,
                'cookie_secret':self.get_from_vault('BokehCookieSecret','Content'),}
        
    @overload
    def get_from_vault(self, account_name: str, components: None) -> dict: ...
    @overload
    def get_from_vault(self, account_name: str, components: str) -> str: ...
    @overload
    def get_from_vault(self, account_name: str, components: list[str]) -> list[str]: ...
    
    def get_from_vault(self, account_name: str, components: Optional[str | list[str]] =None ) \
            -> dict | str | list[str]:
        import json

        # Allowing JMP add-in access to vault for now
        import os
        if os.environ.get("DATAVACUUM_FROM_JMP")=='YES':
            from datavac.util.dvlogging import logger
            logger.warning("JMP add-in is accessing vault.  This makes isolated testing difficult.")
        #    raise Exception("JMP add-in should not assume vault access.  Makes isolated testing difficult.")

        result_text=self._getter(self.api_url,self.appid,self.safe,account_name)
        values=json.loads(result_text)
        try:
            if components is None:
                return values
            elif type(components) is str:
                return values[components]
            else:
                return [values[comp] for comp in components]
        except KeyError:
            raise Exception(f"Failed to find a component among {components} in vault response: {list(values.keys())}")
        
    def _cli_print_account(self,*args):
        """CLI command to get a value from the CyberArk vault."""
        import argparse
        parser = argparse.ArgumentParser(description='Get a value from the CyberArk vault')
        parser.add_argument('account_name', type=str, help='The name of the account in the vault')
        parser.add_argument('--include_secret','-i', action='store_true',
                help='Include the password/secret (ie "Content" field) in the output (default: False)')
        namespace = parser.parse_args(args)
        
        result: dict = self.get_from_vault(namespace.account_name) # type: ignore
        if not namespace.include_secret:
            result.pop('Content', None)
        for k, v in result.items():
            print(f"{k}: {v}")

    def get_cli_funcs(self) -> dict[str, Callable]:
        """Returns a dictionary of CLI functions for CyberArk vault operations."""
        return {
            'clear_cache': self._cli_clear_vault_cache,
            'print_account': self._cli_print_account,
        }

        

def _raw_get_vault_response_text(api_url:str, appid:str, safe: str, account_name: str) -> str:
    import requests
    try:
        from requests_kerberos import HTTPKerberosAuth
    except ImportError:
        raise ImportError("requests-kerberos is required for CyberArk vault access."\
                          "  Please install it via 'pip install requests-kerberos' or by datavacuum's [kerberos_clientside] option.")
    req=f"{api_url}/Accounts?AppID={appid}&Safe={safe}&Object={account_name}"
    with time_it(f"Vault request for {account_name}"):
        result=requests.get(req, auth=HTTPKerberosAuth(),
                            verify=PCONF().cert_depo.get_ssl_rootcert_path_for_vault())
    if result.status_code!=200:
        raise Exception(f"Failed to get from vault: {result.text}")
    return result.text


