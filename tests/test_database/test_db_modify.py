from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from datavac import unload_my_imports
from datavac.trove.mock_trove import MockTrove
from datavac.trove.mock_trove import MockReaderCard

if TYPE_CHECKING:
    import pandas as pd
    from datavac.io.measurement_table import UniformMeasurementTable, MultiUniformMeasurementTable

def make_project_config(num_meas_cols: int, num_extr_cols: int, extr_sign: str = ''):
    """Generates a project configuration for testing purposes.

    Two measurement groups are created:
    - `test_group_nosw`: A measurement group without sweeps, with a specified number of measurement and extraction columns.
    - `test_group_yessw`: A measurement group with sweeps, with one measurement column and one extraction column.
    
    Args:
        num_meas_cols: Number of measurement columns to generate for the test_group_nosw.
        num_extr_cols: Number of extraction columns to generate for the test_group_nosw.
        extr_sign: A string to append to the extraction column values, default is an empty string.
        
    """
    from datavac.appserve.dvsecrets.vaults.demo_vault import DemoVault
    from datavac.config.data_definition import DVColumn, SemiDeviceDataDefinition
    from datavac.config.data_definition import HigherAnalysis
    from datavac.config.project_config import ProjectConfiguration
    from datavac.measurements.measurement_group import MeasurementGroup
    from datavac.util.util import asnamedict
    meas_cols = [DVColumn(f'TestCol{i}', 'string', f'Test Meas column {i}') for i in range(num_meas_cols)]
    extr_cols = [DVColumn(f'ExtrCol{i}', 'string', f'Test Extr column {i}') for i in range(num_extr_cols)]
    @dataclass(eq=False,repr=False)
    class MG(MeasurementGroup):
        def available_extr_columns(self) -> dict[str, DVColumn]:
            return asnamedict(*super().available_extr_columns().values(), *extr_cols,
                              DVColumn('BonusExtrCol0', 'string', 'Bonus Extr column 0'))
        def extract_by_umt(self, measurements: UniformMeasurementTable, other: Optional[MultiUniformMeasurementTable] = None) -> None:
            for col in self.extr_column_names:
                if 'Bonus' in col: continue
                measurements[col]=measurements[col.replace('Extr','Test')]\
                    .apply(lambda x: x.replace('meas','extr')+extr_sign)
            if other is not None:
                for col in self.extr_column_names:
                    if 'Bonus' not in col: continue
                    measurements[col]=other[col.replace('BonusExtr','Extr')]\
                        .apply(lambda x: x+extr_sign)

    def generate(mg_name: str, only_sampleload_info: dict[str, str] = {}, **kwargs) -> list[pd.DataFrame]:
        import pandas as pd
        import numpy as np
        match mg_name:
            case 'test_group_nosw':
                return [pd.DataFrame({'TestCol0':['meas0_col0','meas1_col0'],
                                      'TestCol1':['meas0_col1','meas1_col1']}).convert_dtypes()]
            case 'test_group_yessw':
                return [pd.DataFrame({'RawData':[{'header0':np.r_[1,2,3,4,5].astype('float32')},{'header0':np.r_[2,4,6,8,10].astype('float32')}],
                                      'TestCol0':['meas0_col0','meas1_col0'],
                                      'TestCol1':['meas0_col1','meas1_col1']}).convert_dtypes()]
            case _: raise ValueError(f"Unknown measurement group name: {mg_name}")
    def sample_func():
        return {'sample0': {'SampleName':'sample0','SampleInfoCol0': 'sample0_info0',
                            'SampleInfoCol1': 'sample0_info1', 'MaskSet':'MainMask'},
                'sample1': {'SampleName':'sample1','SampleInfoCol0': 'sample1_info0',
                            'SampleInfoCol1': 'sample1_info1', 'MaskSet':'MainMask'}}
    def anls_func(nosw: pd.DataFrame, yessw: pd.DataFrame) -> pd.DataFrame:
        import pandas as pd
        return pd.DataFrame({'AnlsCol_nosw':   nosw['ExtrCol0'].str.replace('extr','anls'),
                             'AnlsCol_yessw': yessw['ExtrCol0'].str.replace('extr','anls')})
    
    return ProjectConfiguration(deployment_name='datavac_dbtest',
                                data_definition=SemiDeviceDataDefinition(
                                    measurement_groups=asnamedict(
                                    MG(involves_sweeps=False,
                                        name ='test_group_nosw', description ='Test measurement group, no sweeps',
                                        reader_cards={'': [MockReaderCard(reader_func=generate,sample_func=sample_func)]},
                                        meas_columns=meas_cols, extr_column_names=[c.name for c in extr_cols],),
                                    MG(involves_sweeps=True,
                                        name ='test_group_yessw', description ='Test measurement group, yes sweeps',
                                        required_dependencies= {'test_group_nosw':'other'},
                                        reader_cards={'': [MockReaderCard(reader_func=generate,sample_func=sample_func)]},
                                        meas_columns=meas_cols[:1], extr_column_names=[c.name for c in extr_cols][:1]+['BonusExtrCol0'],)),
                                    troves={'': MockTrove()},
                                    higher_analyses=asnamedict(HigherAnalysis('test_anls', 'Test analysis',
                                        analysis_function= anls_func,
                                        analysis_columns=[DVColumn('AnlsCol_nosw', 'string', 'Analysis column for nosw'),
                                                         DVColumn('AnlsCol_yessw', 'string', 'Analysis column for yessw')],
                                        required_dependencies={'test_group_nosw':'nosw','test_group_yessw':'yessw'},)),
                                        
                                ),
                                vault=DemoVault(dbname='datavac_dbtest')
                                )
    


def test_update_meas_groups():

    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.config.data_definition import DDEF
    from datavac.database.db_create import ensure_clear_database, create_all
    from datavac.database.db_modify import update_measurement_group_tables
    from datavac.database.db_util import read_sql
    from datavac.database.db_semidev import upload_mask_info
    from datavac.io.make_diemap import make_fullwafer_diemap
    from datavac.database.db_upload_meas import read_and_enter_data
    from datavac.database.db_modify import heal
    import numpy as np

    # Make a project where the nosw group has 1 meas column and 1 extr column,
    # and the yessw group has 1 meas column and 1 extr column.
    PCONF(make_project_config(1,1))
    ensure_clear_database()
    create_all()
    
    # Check the columns of the nosw group
    assert read_sql('select * from vac."Meas -- test_group_nosw"').columns.tolist()\
        == ['loadid', 'measid', 'rawgroup', 'TestCol0']
    assert read_sql('select * from vac."Extr -- test_group_nosw"').columns.tolist()\
        == ['loadid', 'measid', 'ExtrCol0']
    
    # Populate the database with some data and check
    upload_mask_info({'MainMask': make_fullwafer_diemap(name='deprecatethis', aindex=30, bindex=20, save_csv=False)})
    read_and_enter_data()
    data_back=read_sql('select * from jmp."test_group_nosw" order by loadid,measid')
    assert np.all(data_back['TestCol0']==['meas0_col0','meas1_col0','meas0_col0','meas1_col0'])
    assert np.all(data_back['ExtrCol0']==['extr0_col0','extr1_col0','extr0_col0','extr1_col0'])

    # Now clear out the configuration and reconfigure with 2 meas columns and 1 extr column
    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.database.db_modify import update_measurement_group_tables
    from datavac.database.db_util import read_sql
    PCONF(make_project_config(2,1))

    # Update the measurement group tables, make sure the nosw data is dropped
    # and the yessw group is still there.
    update_measurement_group_tables()
    assert len(read_sql('select * from jmp."test_group_nosw"  order by loadid,measid'))==0
    assert len(read_sql('select * from jmp."test_group_yessw" order by loadid,measid'))>0

    # Now heal and make sure the data is put back
    heal()
    assert len(read_sql('select * from jmp."test_group_yessw" order by loadid,measid'))>0
    assert read_sql('select * from vac."Meas -- test_group_nosw"').columns.tolist()\
        == ['loadid', 'measid', 'rawgroup', 'TestCol0', 'TestCol1']
    assert read_sql('select * from vac."Extr -- test_group_nosw"').columns.tolist()\
        == ['loadid', 'measid', 'ExtrCol0']
    data_back=read_sql('select * from jmp."test_group_nosw" order by loadid,measid')
    assert np.all(data_back['TestCol0']==['meas0_col0','meas1_col0','meas0_col0','meas1_col0'])
    assert np.all(data_back['TestCol1']==['meas0_col1','meas1_col1','meas0_col1','meas1_col1'])
    assert np.all(data_back['ExtrCol0']==['extr0_col0','extr1_col0','extr0_col0','extr1_col0'])
    assert len(read_sql('select * from vac."ReLoad_"'))==0
    assert len(read_sql('select * from vac."ReExtr_"'))==0

    # Now clear out the configuration and reconfigure with 2 meas columns and 2 extr column
    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.database.db_modify import update_measurement_group_tables
    from datavac.database.db_util import read_sql
    PCONF(make_project_config(2,2))
    
    # Update the measurement group tables, make sure the nosw extraction is dropped, but not measurement
    # and the yessw is untouched.
    update_measurement_group_tables()
    assert len(read_sql('select * from vac."Meas -- test_group_nosw"  order by loadid,measid'))>0
    assert len(read_sql('select * from vac."Meas -- test_group_yessw" order by loadid,measid'))>0
    assert len(read_sql('select * from vac."Extr -- test_group_nosw"  order by loadid,measid'))==0
    assert len(read_sql('select * from vac."Extr -- test_group_yessw" order by loadid,measid'))>0

    # Now heal and make sure the data is put back
    heal()
    assert read_sql('select * from vac."Meas -- test_group_nosw"').columns.tolist()\
        == ['loadid', 'measid', 'rawgroup', 'TestCol0', 'TestCol1']
    assert read_sql('select * from vac."Extr -- test_group_nosw"').columns.tolist()\
        == ['loadid', 'measid', 'ExtrCol0', 'ExtrCol1']
    data_back=read_sql('select * from jmp."test_group_nosw" order by loadid,measid')
    assert np.all(data_back['TestCol0']==['meas0_col0','meas1_col0','meas0_col0','meas1_col0'])
    assert np.all(data_back['TestCol1']==['meas0_col1','meas1_col1','meas0_col1','meas1_col1'])
    assert np.all(data_back['ExtrCol0']==['extr0_col0','extr1_col0','extr0_col0','extr1_col0'])
    assert np.all(data_back['ExtrCol1']==['extr0_col1','extr1_col1','extr0_col1','extr1_col1'])
    assert len(read_sql('select * from vac."ReLoad_"'))==0
    assert len(read_sql('select * from vac."ReExtr_"'))==0

    # Set up the config to add a re-extraction sign to the extraction columns
    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.config.data_definition import DDEF
    from datavac.database.db_upload_meas import perform_and_enter_extraction
    from datavac.database.db_get import get_data
    PCONF(make_project_config(2,2,extr_sign='-reextr'))

    # Reextract sample0 for nosw group and check the data
    perform_and_enter_extraction(DDEF().troves[''], samplename='sample0',only_meas_groups=['test_group_nosw'])
    data_back=read_sql('select * from jmp."test_group_nosw" order by loadid,measid')
    assert np.all(data_back['ExtrCol0']==['extr0_col0-reextr','extr1_col0-reextr','extr0_col0','extr1_col0'])
    assert np.all(data_back['ExtrCol1']==['extr0_col1-reextr','extr1_col1-reextr','extr0_col1','extr1_col1'])
    
    # Set up the config to add another re-extraction sign to the extraction columns
    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.config.data_definition import DDEF
    from datavac.database.db_upload_meas import perform_and_enter_extraction
    from datavac.database.db_get import get_data
    PCONF(make_project_config(2,2,extr_sign='-reextr2'))

    # Reextract sample0 for yessw group and check the data
    perform_and_enter_extraction(DDEF().troves[''], samplename='sample0',only_meas_groups=['test_group_yessw'])
    data_back=get_data('test_group_yessw', include_sweeps=True, unstack_headers=True)
    assert np.all(data_back.loc[0,'header0']==np.r_[1.0,2,3,4,5].astype('float32'))
    assert np.all(data_back['ExtrCol0']==['extr0_col0-reextr2','extr1_col0-reextr2','extr0_col0','extr1_col0'])
    assert np.all(data_back['BonusExtrCol0']==['extr0_col0-reextr-reextr2','extr1_col0-reextr-reextr2','extr0_col0','extr1_col0'])

if __name__ == "__main__":
    test_update_meas_groups()