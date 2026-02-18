from __future__ import annotations
from functools import partial
import os
from pathlib import Path
from typing import Mapping
from datavac.config.data_definition import TYPE_CHECKING, DVColumn, SemiDeviceDataDefinition
from datavac.config.layout_params.folder_layout_params import get_folder_layout_params
from datavac.config.project_config import ProjectConfiguration
from datavac.appserve.dvsecrets.vaults.demo_vault import DemoVault
from datavac.examples.example_data import read_csv
from datavac.measurements.logic_cell import InverterDC, OscopeDivider, OscopeFormulaLogic, OscopeRingOscillator
from datavac.measurements.measurement_group import SemiDevMeasurementGroup
from datavac.measurements.transistor import IdVg
from datavac.measurements.capacitor import CapCV
from datavac.trove.classic_folder_trove import ClassicFolderTrove, ClassicFolderTroveReaderCard
from datavac.util.util import asnamedict
from datavac.examples.demo3 import EXAMPLE_DATA_DIR, dbname
if TYPE_CHECKING:
        import pandas as pd

def read_csv_if_not_corrupt(file:Path,*args,**kwargs):
    with open(file,'r') as f:
        first_line=f.readline()
        if 'CORRUPT' in first_line:
            raise ValueError(f"File {file.name} is marked as corrupt")
    return read_csv(file,*args,**kwargs)

def get_project_config() -> ProjectConfiguration:
    def make_reader_cards(glob) -> Mapping[str, list[ClassicFolderTroveReaderCard]]:
        def pr(df: pd.DataFrame): df['Structure']=df['Site']
        rc=ClassicFolderTroveReaderCard(
            glob=glob, reader_func=read_csv_if_not_corrupt, post_reads=[pr],
            read_from_filename_regex= r'^(?P<LotSample>(?P<Lot>[A-Za-z0-9]+)_(?P<Sample>[A-Za-z0-9]+))',
            )
        return {'':[rc]}

    measurement_groups=asnamedict(
        CapCV(name='A_Cap_CV',
            description='Capacitor CV measurements', 
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_A_Cap_CV.csv'),
            layout_param_group='Cap', optional_dependencies={'A_Open_CV':'opens'},
            open_match_cols=['LotSample','layout_style'], connect_to_dies=False,
        ),
        CapCV(name='A_Open_CV',
            description='Open capacitor CV measurements for subtraction from Cap_CV',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_A_Open_CV.csv'),
            layout_param_group='Cap', connect_to_dies=False,
        ),
        CapCV(name='B_Cap_CV',
            description='Capacitor CV measurements', 
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_B_Cap_CV.csv'),
            layout_param_group='Cap', optional_dependencies={'B_Open_CV':'opens'},
            open_match_cols=['LotSample','layout_style'], connect_to_dies=False,
        ),
        CapCV(name='B_Open_CV',
            description='Open capacitor CV measurements for subtraction from Cap_CV',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_B_Open_CV.csv'),
            layout_param_group='Cap', connect_to_dies=False,
        ),
        CapCV(name='C_Cap_CV',
            description='Capacitor CV measurements', 
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_C_Cap_CV.csv'),
            layout_param_group='Cap', optional_dependencies={'C_Open_CV':'opens'},
            open_match_cols=['LotSample','layout_style'], connect_to_dies=False,
        ),
        CapCV(name='C_Open_CV',
            description='Open capacitor CV measurements for subtraction from Cap_CV',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_C_Open_CV.csv'),
            layout_param_group='Cap', connect_to_dies=False,
        ),
    )
    trove=ClassicFolderTrove(
        load_info_columns=[DVColumn('Lot', 'string', 'Lot ID of the sample'),
                           DVColumn('Sample', 'string', 'Sample ID of the sample')],
        read_dir=Path(EXAMPLE_DATA_DIR),
    )
        
    return ProjectConfiguration(
        deployment_name='demo3',
        data_definition=SemiDeviceDataDefinition(
            measurement_groups=measurement_groups,
            get_masks_func=lambda: {},
            sample_identifier_column=DVColumn('LotSample','string','Lot and Sample ID of the sample'),
            troves={'':trove},
            layout_params_func=partial(
                get_folder_layout_params,
                layout_params_dir=Path(__file__).parent.parent/'demo1' / 'layout_params',
                layout_params_yaml=Path(__file__).parent.parent/'demo1' / 'layout_params.yaml',),
            sample_info_completer=(lambda d: dict(**d,**(dict(MaskSet='Mask1') if 'MaskSet' not in d else {})))
        ),
        vault=DemoVault(dbname=dbname),
    )