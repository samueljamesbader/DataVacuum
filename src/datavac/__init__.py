import os
import sys

from importlib import resources as irsc
from dotenv import load_dotenv as _load_dotenv

from .util.logging import logger

__version__='0.0.1'

_load_dotenv()

def unload_my_imports(imports=['datavac','bokeh_transform_utils']):
    modules_to_drop=[k for k in sys.modules if any((i in k for i in imports))]
    if len(modules_to_drop):
        print(f"Unloading {', '.join(sorted(modules_to_drop))}")
    for k in modules_to_drop:
        del sys.modules[k]

# TODO: Move to datavac.util.conf
def load_config_pkg():
    conf_pkg=os.environ.get('DATAVACUUM_CONFIG_PKG',None)
    conf_dir=os.environ.get('DATAVACUUM_CONFIG_DIR',None)
    assert (conf_pkg is not None) or (conf_dir is not None),\
        "Must set one of DATAVACUUM_CONFIG_PKG or DATAVACUUM_CONFIG_DIR"
    if conf_pkg is not None:
        try:
            with irsc.as_file(irsc.files(conf_pkg)) as conf_path:
                assert conf_path.exists(), \
                    f"Config package {conf_pkg} not found at {str(conf_path)}"
                if conf_dir is not None:
                    assert conf_path.samefile(conf_dir),\
                        "DATAVACUUM_CONFIG_PKG and DATAVACUUM_CONFIG_DIR point to different files"
                os.environ['DATAVACUUM_CONFIG_DIR']=str(conf_path)
        except ModuleNotFoundError as e:
            raise Exception(f"Config package {conf_pkg} pointed to by DATAVACUUM_CONFIG_PKG not found") from e
load_config_pkg()