from datavac.io.meta_reader import quick_read_filename

from datavac.tests.freshtestdb import example_data

def test_logic(example_data):
    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_orcell_logic.csv')
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['Site'])==['good_or','bad_or']
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['truth_table_pass'])==[True,False]

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_dff_logic.csv')
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['Site'])==['good_dff1','bad_dff','good_dff2']
    assert list(mt2mg2dat['lot1_sample1']['logic_oscope']['truth_table_pass'])==[True,False,True]

