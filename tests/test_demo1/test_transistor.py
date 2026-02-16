def test_IdVg():
    import numpy as np
    from datavac.trove.trove_util import quick_read_filename
    from datavac.examples.demo1.demo1_example_data import get_transistor

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_nMOS_IdVg.csv')
    print(mt2mg2dat['lot1_sample1']['nMOS_IdVg'][['SS [mV/dec]','Ron [ohm]','RonW [ohm.um]']])

    nmosdat=mt2mg2dat['lot1_sample1']['nMOS_IdVg']
    xtors=[get_transistor(mask='Mask1',structure=s) for s in nmosdat['Site']]
    assert np.allclose(nmosdat['SS [mV/dec]'], [60*xtor.n for xtor in xtors], rtol=0.01)
    assert np.allclose(nmosdat['Ron [ohm]'], [xtor.approximate_Ron([1])[0] for xtor in xtors], rtol=0.01)

    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_pMOS_IdVg.csv')
    print(mt2mg2dat['lot1_sample1']['pMOS_IdVg'][['SS [mV/dec]','Ron [ohm]','RonW [ohm.um]']])

    pmosdat=mt2mg2dat['lot1_sample1']['pMOS_IdVg']
    xtors=[get_transistor(mask='Mask1',structure=s) for s in pmosdat['Site']]
    assert np.allclose(pmosdat['SS [mV/dec]'], [60*xtor.n for xtor in xtors], rtol=0.01)
    assert np.allclose(pmosdat['Ron [ohm]'], [xtor.approximate_Ron([-1])[0] for xtor in xtors], rtol=0.01)

if __name__ == '__main__':
    
    import os
    os.environ["DATAVACUUM_CONTEXT"]="builtin:demo1"
    test_IdVg()