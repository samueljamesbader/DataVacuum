import sys

from dotenv import load_dotenv as _load_dotenv

from datavac.util.conf import config_datavacuum
import datavac.util.logging

__version__='0.0.2'

_load_dotenv()
config_datavacuum()


def unload_my_imports(imports=['datavac','bokeh_transform_utils']):
    modules_to_drop=[k for k in sys.modules if any((i in k for i in imports))]
    if len(modules_to_drop):
        print(f"Unloading {', '.join(sorted(modules_to_drop))}")
    for k in modules_to_drop:
        del sys.modules[k]