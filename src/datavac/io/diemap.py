from typing import cast
import pickle

from datavac.util.dvlogging import logger
from datavac.util.util import returner_context, import_modfunc


_dietabs=None
def get_die_table(mask, conn=None):
    from datavac.database.db_util import read_sql
    global _dietabs
    if _dietabs is None:
        # TODO: This should be replaced with an SQLAlchemy query
        all_dietabs=read_sql(f"""select * from "vac"."Dies" """,conn).convert_dtypes()
        _dietabs={k:v for k,v in all_dietabs.groupby("MaskSet")}
    return _dietabs[mask]

_diegeoms={}
def get_die_geometry(mask, conn=None):
    from sqlalchemy import select
    from datavac.database.db_connect import get_engine_ro
    from datavac.config.data_definition import DDEF
    from datavac.config.data_definition import SemiDeviceDataDefinition
    masktab=cast(SemiDeviceDataDefinition,DDEF()).subsample_references['MaskSet'].dbtable()
    if mask not in _diegeoms:
        with (returner_context(conn) if conn else get_engine_ro().begin()) as conn:
            res=conn.execute(select(masktab.c.info_pickle).where(masktab.c.Mask==mask)).all()
        assert len(res)==1, f"Couldn't get info from database about mask {mask}"
        # Must ensure restricted write access to DB since this allows arbitrary code execution
        _diegeoms[mask]=pickle.loads(res[0][0])
    return _diegeoms[mask]

# TODO: Should this be cached in DB?
_diecrms={}
def get_custom_dieremap(mask, remap_name, conn=None):
    from datavac.config.data_definition import SemiDeviceDataDefinition
    from datavac.config.data_definition import DDEF
    mask_yaml=cast(SemiDeviceDataDefinition,DDEF())._get_mask_yaml()

    if (mask,remap_name) not in _diecrms:
        try: conf=mask_yaml['custom_remaps'][mask][remap_name]
        except KeyError as e:
            raise KeyError(f"Custom remap {remap_name} not found for mask {mask},"\
                           f" options include {list(mask_yaml['custom_remaps'][mask].keys())}") from e
        func=import_modfunc(conf['generator'])
        try:
            _diecrms[(mask,remap_name)]=func(get_die_table(mask,conn=conn),**conf.get('args',{}))
        except Exception as e:
            logger.error(f"Error generating custom die map {remap_name} for mask {mask}")
            raise e
    return _diecrms[(mask,remap_name)]
