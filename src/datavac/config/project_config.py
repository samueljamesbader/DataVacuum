from dataclasses import dataclass, field
from functools import cache
import os
from typing import TYPE_CHECKING, Optional
from pathlib import Path

from datavac.config.cert_depo import CertDepo
from datavac.config.data_definition import DataDefinition
from datavac.appserve.secrets.vault import Vault
import platformdirs

if TYPE_CHECKING: pass

@dataclass
class ProjectConfiguration():
    deployment_name: str
    data_definition: DataDefinition
    vault: Vault = field(default_factory=Vault)
    cert_depo: CertDepo = field(default_factory=CertDepo)

    def __post_init__(self):
        appname=self.deployment_name or 'DEFAULT'
        self.USER_CACHE: Path = (Path(os.environ.get("DATAVACUUM_CACHE_DIR",None) or \
                                    platformdirs.user_cache_path(appname=appname, appauthor='DataVacuum')))/appname

    
_pconf: ProjectConfiguration | None = None
def PCONF() -> ProjectConfiguration:
    """Returns the project configuration singleton."""
    global _pconf
    if _pconf is None:
        #_pconf = ProjectConfiguration()  # Replace with actual project configuration class
        raise NotImplementedError("Haven't figured this out yet.")
    return _pconf

def is_server() -> bool:
    """Returns whether the project is running on a server."""
    return os.environ.get('DATAVACUUM_IS_SERVER','False').lower() in ['true', '1', 'yes']