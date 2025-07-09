from datetime import datetime
import pickle

from datavac.database.db_connect import get_engine_ro, get_engine_so
from datavac.util.dvlogging import time_it
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pgsql_insert, BYTEA, TIMESTAMP

from datavac.util.util import returner_context

def _blobtab():
    from datavac.database.db_structure import DBSTRUCT
    return DBSTRUCT().get_blob_store_dbtable()

def store_obj(name,obj,conn=None):
    with (returner_context(conn) if conn else get_engine_so().begin()) as conn:
        update_info=dict(name=name,blob=pickle.dumps(obj),date_stored=datetime.now())
        conn.execute(pgsql_insert(_blobtab()).values(**update_info) \
                     .on_conflict_do_update(index_elements=['name'],set_=update_info))
def get_obj(name, conn=None):
    with time_it(f"Getting {name} from DB",threshold_time=.1):
        with (returner_context(conn) if conn else get_engine_ro().begin()) as conn:
            res=list(conn.execute(select(_blobtab().c.blob).where(_blobtab().c.name==name)).all())
    if not len(res): raise KeyError(name)
    assert len(res)==1
    with time_it(f"Unpickling {name} from DB",threshold_time=.1):
        print(len(res[0][0]))
        return pickle.loads(res[0][0])
def get_obj_date(name, conn=None):
    with time_it(f"Getting '{name}' date from DB",threshold_time=.001):
        with (returner_context(conn) if conn else get_engine_ro().begin()) as conn:
            res=list(conn.execute(select(_blobtab().c.date_stored).where(_blobtab().c.name==name)).all())
    if not len(res): raise KeyError(name)
    assert len(res)==1
    return res[0][0]