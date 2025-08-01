from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Callable
import argparse
import os

from datavac.database.db_connect import DBConnectionMode

if TYPE_CHECKING:
    from datavac.database.db_connect import PostgreSQLConnectionInfo
    from langchain.chat_models import BaseChatModel


class Vault():
    def get_db_connection_info(self, usermode: DBConnectionMode = DBConnectionMode.READ_ONLY) -> 'PostgreSQLConnectionInfo':
        """Fetches the connection information for the database from the vault."""
        raise NotImplementedError("Subclass should implement")
    
    def get_access_key_sign_seed(self) -> bytes:
        """Fetches the seed for signing access keys from the vault."""
        raise NotImplementedError("Subclass should implement")
    
    def get_auth_info(self) -> dict[str,str]: 
        """Fetches the website authentication information from the vault."""
        raise NotImplementedError("Subclass should implement")
    
    def get_llm_connection_factory(self) -> Callable[...,BaseChatModel]:
        """Fetches the connection secrets for an LLM from the vault, returning an LLM factory function."""
        #raise NotImplementedError("Subclass should implement")

        # Get credentials from environment
        modelname = os.environ.get("DATAVACUUM_LLM_MODEL","gpt-4o")
        base_url = os.environ.get("DATAVACUUM_LLM_BASE_URL")
        proxy = os.environ.get('DATAVACUUM_LLM_PROXY')
        api_key = os.environ.get('DATAVACUUM_LLM_API_KEY')
        if not api_key: raise ValueError("Environment variable 'DATAVACUUM_LLM_API_KEY' is not set.")

        # Special for testing purposes, allow a function dotpath to be specified instead of api_key
        # Not recommended for production
        from datavac.util.util import import_modfunc
        if ':' in api_key: api_key=import_modfunc(api_key)()

        modelclass: BaseChatModel = None
        match modelname:
            case modelname if modelname.startswith('gpt-'):
                from langchain_openai import ChatOpenAI
                modelclass = ChatOpenAI
            case _: raise ValueError(f"Unsupported model name: {modelname}")
        getter = partial(modelclass,
            model=modelname, api_key=api_key, base_url=base_url, openai_proxy=proxy,
            # TODO: remove these parameters 
            temperature=0, timeout=None, max_retries=2)

        return getter
    
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