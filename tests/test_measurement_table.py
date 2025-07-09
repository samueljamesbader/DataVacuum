import numpy as np
import pandas as pd

from datavac.io.measurement_table import MultiUniformMeasurementTable, UniformMeasurementTable
from datavac.io.OLDdatabase import get_database, heal
from datavac.tests.freshtestdb import make_fresh_testdb


def test_mumt():
    from datavac.examples.demo1 import dbname
    make_fresh_testdb(dbname)
    df1=pd.DataFrame({'RawData':[
        {'header1':np.r_[1,2,3],'header2':np.r_[4,5,6]},
        {'header1':np.r_[7,8,9],'header2':np.r_[10,11,12]}],
        'scalar1':[-1,-2],
    })
    df2=pd.DataFrame({'RawData':[
        {'header1':np.r_[13,14,15],},
        {'header1':np.r_[16,17,18],}],
        'scalar1':[-3,-4],
    })
    mumt=MultiUniformMeasurementTable.from_read_data([df1,df2],None,None)

    db=get_database()
    db.push_data({'LotSample':'lot2_sample1','Lot':'lot2','Sample':'sample1','Mask':'Mask1'},{'misc_test':mumt},)
    mtback=db.get_data_for_regen('misc_test','lot2_sample1',)
    dfback=mtback._dataframe
    assert mtback._umts[0].headers==['header1','header2']
    assert mtback._umts[1].headers==['header1']
    assert list(dfback['scalar1'])==[-1,-2,-3,-4]
    with db.engine_begin() as conn:
        db.dump_extractions('misc_test',conn)
    heal(db)
    mtback=db.get_data_for_regen('misc_test','lot2_sample1',)
    dfback=mtback._dataframe
    assert mtback._umts[0].headers==['header1','header2']
    assert mtback._umts[1].headers==['header1']
    assert list(dfback['scalar1'])==[-1,-2,-3,-4]
