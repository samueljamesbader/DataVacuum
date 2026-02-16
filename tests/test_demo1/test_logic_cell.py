def test_logic():
    import numpy as np
    from datavac.examples.demo1.demo1_example_data import get_ring, get_inverter
    from datavac.trove.trove_util import quick_read_filename

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_invs.csv')
    assert list(mt2mg2dat['lot1_sample1']['inverter_DC']['Site'])==['inv1','inv2']
    invs=[get_inverter(mask='Mask1',structure=s) for s in mt2mg2dat['lot1_sample1']['inverter_DC']['Site']]
    #assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['Vmid [V]'],[inv.Vmid for inv in invs],rtol=.01)
    assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['max_gain'],[inv.gain for inv in invs],rtol=.01)
    ics=[inv.characteristics() for inv in invs]
    assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['VIL [V]'],[ic['V_IL'] for ic in ics],rtol=.01)
    assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['VIH [V]'],[ic['V_IH'] for ic in ics],rtol=.01)
    assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['VOL [V]'],[ic['V_OL'] for ic in ics],rtol=.01)
    assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['VOH [V]'],[ic['V_OH'] for ic in ics],rtol=.01)
    assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['NML [V]'],[ic['NML'] for ic in ics],rtol=.01)
    assert np.allclose(mt2mg2dat['lot1_sample1']['inverter_DC']['NMH [V]'],[ic['NMH'] for ic in ics],rtol=.01)

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
    print("passed")

if __name__ == '__main__':
    import os
    os.environ["DATAVACUUM_CONTEXT"]="builtin:demo1"
    from conftest import _example_data
    _example_data()
    test_logic()