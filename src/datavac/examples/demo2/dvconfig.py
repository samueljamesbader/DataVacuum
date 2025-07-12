from __future__ import annotations
from functools import cache
from typing import TYPE_CHECKING, Any, cast
from datavac.appserve.dvsecrets.vaults.demo_vault import DemoVault
from datavac.config.data_definition import DVColumn, SemiDeviceDataDefinition
from datavac.config.layout_params import LayoutParameters
from datavac.config.project_config import ProjectConfiguration
from datavac.examples.demo2 import EXAMPLE_DATA_DIR, dbname
from datavac.measurements.transistor import IdVg
from datavac.trove.mock_trove import MockReaderCard, MockTrove
from datavac.util.util import asnamedict
import io

if TYPE_CHECKING:
    import pandas as pd

@cache
def get_split_table():
    import pandas as pd
    split_table = pd.read_csv(
        io.StringIO("""
        LotSample,     MaskSet,     VTSkew,   CDSkew
        lot1_sample1, MainMask,   low,      1.0
        lot1_sample2, MainMask,   nom,      1.0
        lot1_sample3, MainMask,  high,      1.0
        lot2_sample1, MainMask,   nom,      0.9
        lot2_sample2, MainMask,   nom,      1.0
        lot2_sample3, MainMask,   nom,      1.1
        lot3_sample1, MainMask,   low,      0.9
        lot3_sample2, MainMask,  high,      1.1
        """),skipinitialspace=True).convert_dtypes().set_index('LotSample',verify_integrity=True)
    return split_table

def generate(LotSample: str, mg_name: str) -> list[pd.DataFrame]:
    import pandas as pd
    from datavac.examples.data_mock.mock_devices import Transistor4T
    Lnom= 1e-6  # Nominal length in meters
    dbdf= get_masks()['MainMask'][0]
    split_table=get_split_table()
    
    data=[]
    for _,dbdf_row in dbdf.iterrows():
        radius = dbdf_row['DieRadius [mm]']

        t=Transistor4T(
            VT0={'low':.3,'nom':.4,'high':.5}[split_table.loc[LotSample, 'VTSkew']]+.1*radius/150, # type: ignore
            L=Lnom*split_table.loc[LotSample, 'CDSkew'], # type: ignore
            n=1+(.2/split_table.loc[LotSample, 'CDSkew']), # type: ignore
        )
        data.append({'RawData': {k:v.astype('float32') for k,v in t.generate_potential_idvg(VDs=[.05,1],VGrange=[0,1]).items()},
                     'Structure': 'nMOS1','DieXY': dbdf_row['DieXY'],})
    return [pd.DataFrame(data).convert_dtypes()]
def sample_func() -> dict[str,dict[str,Any]]:
    from datavac.config.project_config import PCONF
    SAMPLENAME_COL=PCONF().data_definition.SAMPLE_COLNAME
    tab=get_split_table().copy()
    tab[tab.index.name] = tab.index
    ret= tab[[c for c in PCONF().data_definition.ALL_SAMPLE_COLNAMES if c in tab]].to_dict(orient='index') # type: ignore
    print(ret)
    return ret # type: ignore


class MockLayoutParams(LayoutParameters):
    def __init__(self):
        import pandas as pd
        self._tables_by_meas: dict[str, pd.DataFrame] = {
            'IdVg': pd.DataFrame({
                'Structure': ['nMOS1'],
                'W [um]': [1],
            }).set_index('Structure', verify_integrity=True),} 
class SemiDeviceDataDefinitionFakeLayout(SemiDeviceDataDefinition):
    def get_flow_names(self) -> list[str]:
        return ['MainFlow']
    def get_split_table_columns(self, flow_name: str) -> list[DVColumn]:
        #from datavac.database.db_structure import pd_to_sql_types
        return [DVColumn(c,dtype.name,c) for c,dtype in get_split_table().dtypes.items()] # type: ignore

def get_masks():
    from datavac.io.make_diemap import make_fullwafer_diemap
    #return {'MainMask':dict(generator='datavac.io.make_diemap:make_fullwafer_diemap',
    #                             args=dict(name='deprecatethis',aindex=30,bindex=20,save_csv=False))}
    return {'MainMask': make_fullwafer_diemap(name='deprecatethis', aindex=30, bindex=20, save_csv=False)}

def get_project_config() -> ProjectConfiguration:
    from datavac.measurements.measurement_group import SemiDevMeasurementGroup
    return ProjectConfiguration(
        deployment_name='datavac_demo2',
        data_definition=SemiDeviceDataDefinitionFakeLayout(
            sample_identifier_column=DVColumn('LotSample','string','Lot and Sample identifier'),
            measurement_groups=asnamedict(
                IdVg(
                    name='IdVg', norm_column='W [um]',
                    description='Id-Vg transfer curves',
                    reader_cards={'':[MockReaderCard(reader_func=generate,sample_func=sample_func)]},
                    meas_columns=[],
                    only_extr_columns=['SS [mV/dec]'],#, 'VTcc1_lin', 'VTcc1_sat'],
                    layout_param_group='IdVg'
                    )
            ),
            layout_params_func=MockLayoutParams,
            get_masks_func=get_masks,
            troves={'': MockTrove()},
        ),
        vault=DemoVault(dbname=dbname),
    )

if __name__ == '__main__':

    from datavac.config.project_config import PCONF
    from datavac.database.db_create import ensure_clear_database, create_all
    from datavac.database.db_semidev import upload_mask_info
    from datavac.database.db_upload_meas import upload_measurement, upload_extraction
    from datavac.database.db_upload_other import upload_sample_descriptor, upload_subsample_reference
    from datavac.database.db_upload_meas import read_and_enter_data

    ddef=cast(SemiDeviceDataDefinitionFakeLayout,PCONF().data_definition)

    ensure_clear_database()
    create_all()
    #upload_mask_info(get_masks())

    upload_sample_descriptor('SplitTable MainFlow', get_split_table())
    #upload_subsample_reference('LayoutParams -- IdVg',ddef.get_layout_params_table('IdVg').reset_index(drop=False))
    read_and_enter_data()
        
    from datavac.database.db_util import read_sql
    #print(read_sql("""select * from vac."Loads_" """))
    #print(read_sql("""select * from vac."Meas -- IdVg" """))
    #print(read_sql("""select * from vac."ReLoad_" """))
    #print(read_sql("""select * from vac."Extr -- IdVg" """))
    print(read_sql("""select * from jmp."IdVg" """))


    #for sample, data_by_mg in sample_to_mg_to_data.items():
    #    upload_measurement(trove, sample_to_sampleloadinfo[sample], data_by_mg)
    #print(read_sql("""select * from vac."ReLoad_" """))