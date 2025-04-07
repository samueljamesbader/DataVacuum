import numpy as np

from datavac.examples.demo1.example_data import get_ring
from datavac.io.meta_reader import quick_read_filename

from datavac.tests.freshtestdb import example_data

def test_logic(example_data):
    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_tiehi_logic.csv')
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['Site'])==['good_tihi','bad_tihi']
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['truth_table_pass'])==[True,False]

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_orcell_logic.csv')
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['Site'])==['good_or','bad_or']
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['truth_table_pass'])==[True,False]

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_dff_logic.csv')
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['Site'])==['good_dff1','bad_dff','good_dff2']
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['truth_table_pass'])==[True,False,True]

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_ros.csv')
    assert list(mt2mg2dat['lot1_sample1']['ROs']['Site'])==['ro1','ro2','ro3','ro4']
    ros=[get_ring(mask='Mask1',structure=s) for s in mt2mg2dat['lot1_sample1']['ROs']['Site']]
    assert np.allclose(mt2mg2dat['lot1_sample1']['ROs']['t_stage [ps]'],[ro.t_stage*1e12 for ro in ros],rtol=.01)

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_divs.csv')
    assert list(mt2mg2dat['lot1_sample1']['divider']['Site'])==['good_div2','bad_div2','good_div4','bad_div4']
    assert list(mt2mg2dat['lot1_sample1']['divider']['correct_division'])==[True,False,True,False]
