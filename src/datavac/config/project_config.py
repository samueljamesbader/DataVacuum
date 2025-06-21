from dataclasses import dataclass, field
from functools import cache
from importlib import import_module
import os
from typing import TYPE_CHECKING, Callable, Optional
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
def PCONF(inject_configuration:Optional[ProjectConfiguration]=None) -> ProjectConfiguration:
    """Returns the project configuration singleton by importing it from the configured module or path.

    Normal usage will check the environment variables DATAVACUUM_CONFIG_MODULE and DATAVACUUM_CONFIG_PATH.
    (Only one of these should be set.)  The module or path should contain a function 'get_project_config'
    that returns a ProjectConfiguration instance.

    Args:
        inject_configuration: If provided, this configuration will be used instead of checking the
        environment variables.  This is mainly useful for testing, not recommended practice.
    """
    global _pconf
    if inject_configuration is not None:
        assert _pconf is None, "Project configuration already initialized"
        _pconf = inject_configuration
    if _pconf is None:
        conf_mod=os.environ.get('DATAVACUUM_CONFIG_MODULE',None)
        conf_pth=os.environ.get('DATAVACUUM_CONFIG_PATH',None)
        assert (conf_mod is not None) != (conf_pth is not None), \
            "Must set one of DATAVACUUM_CONFIG_MODULE or DATAVACUUM_CONFIG_PATH"
        if conf_mod is not None:
            try: mod = import_module(conf_mod)
            except ModuleNotFoundError as e:
                raise Exception(f"Config module {conf_mod} not found") from e
        if conf_pth is not None:
            conf_pth = Path(conf_pth)
            assert conf_pth.exists(), f"Config path {conf_pth} does not exist"
            assert conf_pth.is_file(), f"Config path {conf_pth} is not a file"
            mod = import_module(conf_pth.stem)
        try: func: Callable[[],ProjectConfiguration] = getattr(mod, 'get_project_config')
        except AttributeError:
            raise Exception(f"Module {conf_mod or conf_pth} does not have a 'get_project_config' function")
        _pconf = func()
    return _pconf

def is_server() -> bool:
    """Returns whether the project is running on a server."""
    return os.environ.get('DATAVACUUM_IS_SERVER','False').lower() in ['true', '1', 'yes']