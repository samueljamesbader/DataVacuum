from __future__ import annotations
from functools import cache
from typing import TYPE_CHECKING, Any, cast
from datavac.appserve.dvsecrets.vaults.demo_vault import DemoVault
from datavac.config.data_definition import DVColumn, SemiDeviceDataDefinition
from datavac.config.layout_params import LP, LayoutParameters
from datavac.config.project_config import ProjectConfiguration
from datavac.config.sample_splits import DictSampleSplitManager
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

def completer(partial_sampleload_info: dict[str, Any]) -> dict[str, Any]:
    partial_sampleload_info = partial_sampleload_info.copy()
    if 'LotSample' not in partial_sampleload_info:
        if 'Lot' in partial_sampleload_info and 'Sample' in partial_sampleload_info:
            partial_sampleload_info['LotSample'] = f"{partial_sampleload_info['Lot']}_{partial_sampleload_info['Sample']}"
    else:
        if 'Lot' not in partial_sampleload_info:
            partial_sampleload_info['Lot'] = partial_sampleload_info['LotSample'].split('_')[0]
        if 'Sample' not in partial_sampleload_info:
            partial_sampleload_info['Sample'] = partial_sampleload_info['LotSample'].split('_')[1]
    return partial_sampleload_info

def generate(LotSample: str, mg_name: str) -> list[pd.DataFrame]:
    import pandas as pd
    from datavac.examples.data_mock.mock_devices import Transistor4T
    dbdf= get_masks()['MainMask'][0]
    split_table=get_split_table()
    
    data=[]
    for _,dbdf_row in dbdf.iterrows():
        radius = dbdf_row['DieRadius [mm]']

        for structure,layinfo in LP()._tables_by_meas['IdVg'].iterrows():
            Lnom = layinfo['L [um]']*1e-6
            Wnom = layinfo['W [um]']*1e-6
            t=Transistor4T(
                VT0={'low':.3,'nom':.4,'high':.5}[split_table.loc[LotSample, 'VTSkew']]+.1*radius/150, # type: ignore
                L=Lnom*split_table.loc[LotSample, 'CDSkew'], # type: ignore
                n=1+(.2/split_table.loc[LotSample, 'CDSkew']), # type: ignore
                W=Wnom
            )
            data.append({'RawData': {k:v.astype('float32') for k,v in t.generate_potential_idvg(VDs=[.05,1],VGrange=[0,1]).items()},
                         'Structure': structure,'DieXY': dbdf_row['DieXY'],})
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
    def __init__(self, force_regenerate:bool=False):
        import pandas as pd
        self._tables_by_meas: dict[str, pd.DataFrame] = {
            'IdVg': pd.DataFrame({
                'Structure': ['nMOS1','nMOS2'],
                'W [um]': [1,2],
                'L [um]': [1,1],
            }).set_index('Structure', verify_integrity=True),} 

def get_masks():
    from datavac.io.make_diemap import make_fullwafer_diemap
    #return {'MainMask':dict(generator='datavac.io.make_diemap:make_fullwafer_diemap',
    #                             args=dict(name='deprecatethis',aindex=30,bindex=20,save_csv=False))}
    return {'MainMask': make_fullwafer_diemap(name='deprecatethis', aindex=30, bindex=20, save_csv=False)}

def get_project_config() -> ProjectConfiguration:
    from datavac.measurements.measurement_group import SemiDevMeasurementGroup
    return ProjectConfiguration(
        deployment_name='datavac_demo2',
        data_definition=SemiDeviceDataDefinition(
            sample_identifier_column=DVColumn('LotSample','string','Lot and Sample identifier'),
            sample_info_columns=[
                DVColumn('Lot', 'string', 'Lot name'),
                DVColumn('Sample', 'string', 'Sample name'),],
            split_manager=DictSampleSplitManager(split_tables={'MainFlow': get_split_table().reset_index()}),
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
            sample_info_completer= completer,
            layout_params_func=MockLayoutParams,
            get_masks_func=get_masks,
            troves={'': MockTrove()},
        ),
        vault=DemoVault(dbname=dbname),
    )