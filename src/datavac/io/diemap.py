import os
from pathlib import Path

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#from matplotlib import patches

import pickle

from datavac.io.database import get_database
from datavac.util.conf import CONFIG
from datavac.util.logging import logger
from datavac.util.tables import check_dtypes
from datavac.io.make_diemap import make_fullwafer_diemap as standalone_mfd
from datavac.util.util import returner_context, import_modfunc

#def make_fullwafer_diemap(name, xindex, yindex, radius=150, notchsize=5, plot=False, save=True):
#    return standalone_mfd(name,xindex=xindex,yindex=yindex,
#                   radius=radius,notchsize=notchsize,plot=plot,save_csv=save,
#                   labeller=(lambda x,y:f"L{x:+d}B{y:+d}"))

_dietabs=None
def get_die_table(mask, conn=None):
    global _dietabs
    if _dietabs is None:
        db=get_database(populate_metadata=False)
        with (returner_context(conn) if conn else db.engine.connect()) as conn:
            all_dietabs=pd.read_sql(f"""select * from "vac"."Dies" """,conn).convert_dtypes()
        _dietabs={k:v for k,v in all_dietabs.groupby("Mask")}
    return _dietabs[mask]

_diegeoms={}
def get_die_geometry(mask, conn=None):
    if mask not in _diegeoms:
        _diegeoms[mask]=get_database().get_mask_info(mask,conn=conn)
    return _diegeoms[mask]

# TODO: Should this be cached in DB?
_diecrms={}
def get_custom_dieremap(mask, remap_name, conn=None):
    if (mask,remap_name) not in _diecrms:
        try: conf=CONFIG['custom_remaps'][mask][remap_name]
        except KeyError as e:
            raise KeyError(f"Custom remap {remap_name} not found for mask {mask},"\
                           f" options include {list(CONFIG['custom_remaps'][mask].keys())}") from e
        func=import_modfunc(conf['generator'])
        _diecrms[(mask,remap_name)]=func(get_die_table(mask,conn=conn),**conf.get('args',{}))
    return _diecrms[(mask,remap_name)]
