def test_CV():
    import numpy as np
    from datavac.trove.trove_util import quick_read_filename
    from datavac.examples.example_data import get_capacitor

    mt2mg2dat, mg2ml = quick_read_filename('lot1')
    capdat = mt2mg2dat['lot1_sample1']['Cap_CV']
    expected = [r['C_target'] for _,r in capdat.scalar_table_with_layout_params().iterrows()]
    measured = capdat['Cmean [F]']
    assert np.allclose(measured, expected, rtol=0.01, atol=0)

    expected = [r['C_route'] for _,r in capdat.scalar_table_with_layout_params().iterrows()]
    measured = capdat['Copen [F]']
    assert not np.allclose(measured, 0, rtol=0.01, atol=0)
    assert np.allclose(measured, expected, rtol=0.01, atol=0)

if __name__ == '__main__':
    import os
    os.environ["DATAVACUUM_CONTEXT"] = "builtin:demo1"
    from conftest import _example_data
    _example_data()
    test_CV()
