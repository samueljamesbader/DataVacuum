from datavac import unload_my_imports


from datavac.trove import ReaderCard, Trove
from datavac.config.data_definition import DVColumn
class MockTrove(Trove):
    load_info_columns:list[DVColumn] = []
class MockReaderCard(ReaderCard):
    pass

def make_project_config(num_meas_cols):
    from datavac.appserve.secrets.vault.demo_vault import DemoVault
    from datavac.config.data_definition import DVColumn, SemiDeviceDataDefinition
    from datavac.config.project_config import ProjectConfiguration
    from datavac.measurements import MeasurementGroup
    meas_cols = [DVColumn(f'TestCol{i}', 'float', f'Test column {i}') for i in range(num_meas_cols)]
    return ProjectConfiguration('datavac_dbtest',
                                data_definition=SemiDeviceDataDefinition(
                                    measurement_groups={
                                        'test_group': MeasurementGroup(
                                            name='test_group',
                                            description='Test measurement group',
                                            meas_columns=meas_cols,
                                            extr_column_names=[],#['ExtrCol1', 'ExtrCol2'],
                                            subsample_reference_names=[],
                                            reader_cards={'': [MockReaderCard()]}
                                        ),
                                    },
                                    layout_params_dir=None, # type: ignore
                                    layout_params_yaml=None, # type: ignore
                                    troves={'': MockTrove()}
                                ),
                                vault=DemoVault(dbname='datavac_dbtest')
                                )
    


def test_update_meas_groups():

    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.database.db_create import ensure_clear_database, create_all
    from datavac.database.db_modify import update_measurement_group_tables
    from datavac.database.db_util import read_sql
    PCONF(make_project_config(1))
    ensure_clear_database()
    create_all()
    assert read_sql('select * from vac."Meas -- test_group"').columns.tolist() == ['loadid', 'measid', 'rawgroup', 'TestCol0']

    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.database.db_modify import update_measurement_group_tables
    from datavac.database.db_util import read_sql
    PCONF(make_project_config(2))
    update_measurement_group_tables()
    assert read_sql('select * from vac."Meas -- test_group"').columns.tolist() == ['loadid', 'measid', 'rawgroup', 'TestCol0', 'TestCol1']

if __name__ == "__main__":
    test_update_meas_groups()