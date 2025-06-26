from dataclasses import dataclass

from datavac import unload_my_imports
from datavac.trove.mock_trove import MockTrove
from datavac.trove.mock_trove import MockReaderCard

def make_project_config(num_meas_cols, num_extr_cols):
    from datavac.appserve.secrets.vault.demo_vault import DemoVault
    from datavac.config.data_definition import DVColumn, SemiDeviceDataDefinition
    from datavac.config.project_config import ProjectConfiguration
    from datavac.measurements.measurement_group import MeasurementGroup
    from datavac.util.util import asnamedict
    meas_cols = [DVColumn(f'TestCol{i}', 'float', f'Test Meas column {i}') for i in range(num_meas_cols)]
    extr_cols = [DVColumn(f'ExtrCol{i}', 'float', f'Test Extr column {i}') for i in range(num_extr_cols)]
    @dataclass
    class MG(MeasurementGroup):
        def available_extr_columns(self) -> dict[str, DVColumn]:
            return asnamedict(*super().available_extr_columns().values(),* extr_cols)
    
    return ProjectConfiguration(deployment_name='datavac_dbtest',
                                data_definition=SemiDeviceDataDefinition(
                                    measurement_groups={'test_group': MG(
                                        name ='test_group',
                                        description ='Test measurement group',
                                        reader_cards={'': [MockReaderCard()]},
                                        meas_columns=meas_cols,
                                        extr_column_names=[c.name for c in extr_cols],)},
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
    PCONF(make_project_config(1,1))
    ensure_clear_database()
    create_all()
    assert read_sql('select * from vac."Meas -- test_group"').columns.tolist()\
        == ['loadid', 'measid', 'rawgroup', 'TestCol0']
    assert read_sql('select * from vac."Extr -- test_group"').columns.tolist()\
        == ['loadid', 'measid', 'ExtrCol0']

    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.database.db_modify import update_measurement_group_tables
    from datavac.database.db_util import read_sql
    PCONF(make_project_config(2,1))
    update_measurement_group_tables()
    assert read_sql('select * from vac."Meas -- test_group"').columns.tolist()\
        == ['loadid', 'measid', 'rawgroup', 'TestCol0', 'TestCol1']
    assert read_sql('select * from vac."Extr -- test_group"').columns.tolist()\
        == ['loadid', 'measid', 'ExtrCol0']

    unload_my_imports()
    from datavac.config.project_config import PCONF
    from datavac.database.db_modify import update_measurement_group_tables
    from datavac.database.db_util import read_sql
    PCONF(make_project_config(2,2))
    update_measurement_group_tables()
    assert read_sql('select * from vac."Meas -- test_group"').columns.tolist()\
        == ['loadid', 'measid', 'rawgroup', 'TestCol0', 'TestCol1']
    assert read_sql('select * from vac."Extr -- test_group"').columns.tolist()\
        == ['loadid', 'measid', 'ExtrCol0', 'ExtrCol1']

if __name__ == "__main__":
    test_update_meas_groups()