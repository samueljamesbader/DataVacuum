from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Optional, cast

from datavac.util.logging import logger
from datavac.util.util import returner_context

if TYPE_CHECKING:
    from sqlalchemy import Connection

# TODO: This function is ported from old framework
def upload_mask_info(mask_info: dict[str, Any],conn: Optional[Connection]=None):
    from sqlalchemy.dialects.postgresql import insert as pgsql_insert
    from sqlalchemy import select
    import pandas as pd
    from datavac.util.util import import_modfunc
    from datavac.database.db_structure import DBSTRUCT
    from datavac.database.postgresql_upload_utils import upload_csv
    from datavac.database.db_connect import get_engine_rw
    
    with (returner_context(conn) if conn else get_engine_rw().begin()) as conn:
        masktab = DBSTRUCT().get_sample_reference_dbtable('MaskSet')
        diemtab = DBSTRUCT().get_subsample_reference_dbtable('Dies')
        if not len(mask_info): return
        diemdf=[]
        for mask,info in mask_info.items():
            #dbdf,to_pickle=import_modfunc(info['generator'])(**info['args'])
            dbdf, to_pickle = info
            diemdf.append(dbdf.assign(MaskSet=mask)[[c.name for c in diemtab.columns if c.name!='dieid']])
            update_info=dict(MaskSet=mask,info_pickle=pickle.dumps(to_pickle))
            conn.execute(pgsql_insert(masktab).values(**update_info)\
                         .on_conflict_do_update(index_elements=['MaskSet'],set_=update_info))

        diemdf=pd.concat(diemdf).reset_index(drop=True).reset_index(drop=False)
        previous_dietab=pd.read_sql(select(*diemtab.columns).order_by(diemtab.c['dieid']),conn).reset_index(drop=False)
        # This checks that nothing has changed in the previous table
        # very important to check that because all the measured data is only associated with a die index,
        # so if we accidentally change the die index, even by uploading the tables in a different order...
        # poof all the old data is now associated with the wrong dies or even wrong masks!!
        #print("\n\n\nPREVIOUS:")
        #print(previous_dietab)
        #print("\n\n\nNEW:")
        #print(diemdf)
        #print("\n")
        assert len(previous_dietab.merge(diemdf))==len(previous_dietab),\
            "Can't add to die tables without messing up existing dies"
        upload_csv(diemdf.iloc[len(previous_dietab):].rename(columns={'index':'dieid'}),conn,DBSTRUCT().int_schema,'Dies')

        #print("Successful")


#def update_layout_parameters(layout_params, layout_param_group, conn, dump_extractions_and_analyses=True):


