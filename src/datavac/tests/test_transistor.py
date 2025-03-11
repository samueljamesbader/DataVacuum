import os

import pytest

from datavac.io.meta_reader import quick_read_filename


@pytest.fixture(scope='session')
def example_data():
    from datavac.examples.demo1.example_data import make_example_data
    make_example_data()

def test_IdVg(example_data):
    mt2mg2dat,mg2ml=quick_read_filename('lot1/lot1_sample1_IdVg.csv')
    print(mt2mg2dat['lot1_sample1']['nMOS_IdVg']['SS [mV/dec]'])
