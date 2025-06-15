import argparse
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Callable
from datavac.appserve.secrets.vault import Vault
from datavac.config.project_config import PCONF
from datavac.util.caching import pickle_cached
from datavac.util.logging import time_it
import requests


if TYPE_CHECKING:
    from datavac.database.db_connect import PostgreSQLConnectionInfo

@dataclass
class CyberArkVault():

    api_url: str
    appid: str
    safe: str
    cache_on_disk: bool = False

    _getter: Callable = field(init=False, repr=False)
    def __post_init__(self):
        func = _raw_get_vault_response_text
        if self.cache_on_disk:
            func= pickle_cached(cache_dir=USER_CACHE/"vault",namer=lambda api_url,appid,safe,account_name: f"{appid}_{safe}_{account_name}.pkl")(func)
        self._getter = cache(func)

    def get_db_connection_info(self, usermode: str = 'ro') -> 'PostgreSQLConnectionInfo':
        """Fetches the connection information for the database from the CyberArk vault.
        
        # TODO: This assumption should be removed or simplified
        Assumes the name of the secret in the vault is '{deployment_name}-read' or '{deployment_name}-super' depending on the usermode.
        """
        deployment_name=PCONF().deployment_name
        return PostgreSQLConnectionInfo(**dict(zip(
            ['Uid','Password','Server','Port','Database'],
            self.get_from_vault(f'{deployment_name}-{ {"ro":"read","so":"super"}[usermode]}',['UserName','Content','Address','Port','Database']))))
    
    
    def get_from_vault(self, account_name,components,cached_vault=None,env=None):
        import json
        import os
        if os.environ.get("DATAVACUUM_FROM_JMP")=='YES':
            raise Exception("JMP add-in should not assume vault access.  Makes isolated testing difficult.")

        result_text=self._getter(self.api_url,self.appid,self.safe,account_name)
        values=json.loads(result_text)
        try:
            if type(components) is str:
                return values[components]
            else:
                return [values[comp] for comp in components]
        except KeyError:
            raise Exception(f"Failed to find a component among {components} in vault response: {list(values.keys())}")

def _raw_get_vault_response_text(api_url:str, appid:str, safe: str, account_name: str) -> str:
    from requests_kerberos import HTTPKerberosAuth
    req=f"{api_url}/Accounts?AppID={appid}&Safe={safe}&Object={account_name}"
    with time_it(f"Vault request for {account_name}"):
        result=requests.get(req, auth=HTTPKerberosAuth(), verify=PCONF().cert_depo.get_ssl_rootcert_path_for_vault())
    if result.status_code!=200:
        raise Exception(f"Failed to get from vault: {result.text}")
    return result.text


def cli_clear_vault_cache(*args):
    parser=argparse.ArgumentParser(description='Clear local vault cache')
    namespace=parser.parse_args(args)
    for pth in (USER_CACHE/"vault").glob("*.pkl"):
        pth.unlink()
    print("Cleared vault cache")
