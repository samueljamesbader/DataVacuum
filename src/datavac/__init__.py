import sys
from dotenv import load_dotenv as _load_dotenv

from datavac.config.contexts import load_environment_from_context
import datavac.util.logging

__version__='0.0.2'

_load_dotenv()
load_environment_from_context()


def unload_my_imports(imports=['datavac','bokeh_transform_utils'], silent=False):
    modules_to_drop=[k for k in sys.modules if any((i in k for i in imports))]
    if (not silent) and len(modules_to_drop):
        print(f"Unloading {', '.join(sorted(modules_to_drop))}")
    for k in modules_to_drop:
        del sys.modules[k]

class ThisModule(sys.modules[__name__].__class__):
    @classmethod
    @property
    def logger(cls):
        return datavac.util.logging.logger

    @classmethod
    @property
    def db(cls) -> 'datavac.io.database.Database':
        from datavac.io.database import get_database, PostgreSQLDatabase
        return get_database()
sys.modules[__name__].__class__=ThisModule