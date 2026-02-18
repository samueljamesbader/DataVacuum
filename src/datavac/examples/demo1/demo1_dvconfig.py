from functools import partial
import os
from pathlib import Path
from typing import Mapping
from datavac.config.data_definition import DVColumn, SemiDeviceDataDefinition
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
from datavac.examples.demo1 import EXAMPLE_DATA_DIR, dbname


def get_project_config() -> ProjectConfiguration:
    def make_reader_cards(glob) -> Mapping[str, list[ClassicFolderTroveReaderCard]]:
        rc=ClassicFolderTroveReaderCard(
            glob=glob, reader_func=read_csv,
            read_from_filename_regex= r'^(?P<LotSample>(?P<Lot>[A-Za-z0-9]+)_(?P<Sample>[A-Za-z0-9]+))',
            )
        return {'':[rc]}

    measurement_groups=asnamedict(
        IdVg(name='nMOS_IdVg',
            description='nMOS Id-Vg measurements', norm_column='W [um]',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            only_extr_columns=['SS [mV/dec]', 'RonW [ohm.um]', 'Ron [ohm]'],
            reader_cards=make_reader_cards('*_nMOS_IdVg.csv'),
            layout_param_group='IdVg', pol='n',
        ),
        IdVg(name='pMOS_IdVg',
            description='pMOS Id-Vg measurements', norm_column='W [um]',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            only_extr_columns=['SS [mV/dec]', 'RonW [ohm.um]', 'Ron [ohm]'],
            reader_cards=make_reader_cards('*_pMOS_IdVg.csv'),
            layout_param_group='IdVg', pol='p',
        ),
        CapCV(name='Cap_CV',
            description='Capacitor CV measurements', 
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_Cap_CV.csv'),
            layout_param_group='Cap', optional_dependencies={'Open_CV':'opens'},
            open_match_cols=['layout_style'],
        ),
        CapCV(name='Open_CV',
            description='Open capacitor CV measurements for subtraction from Cap_CV',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            reader_cards=make_reader_cards('*_Open_CV.csv'),
            layout_param_group='Cap',
        ),
        InverterDC(name='inverter_DC',
            description='Inverter DC measurements',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            only_extr_columns=['max_gain'],
            reader_cards=make_reader_cards('*_invs.csv'),
            layout_param_group='InverterDC',
        ),
        OscopeFormulaLogic(name='logic_oscope',
            description='Logic Oscope measurements',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            only_extr_columns=['truth_table_pass'],
            reader_cards=make_reader_cards('*_logic.csv'),
            layout_param_group='Logic',
        ),
        OscopeRingOscillator(name='ROs',
            description='Oscope Ring Oscillator measurements',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            only_extr_columns=['t_stage [ps]'],
            reader_cards=make_reader_cards('*_ros.csv'),
            layout_param_group='ROs',
            stages_col='stages', div_by_col='div_by',
        ),
        OscopeDivider(name='divider',
            description='Oscope Divider measurements',
            meas_columns=[DVColumn('FileName','str','File name of the measurement'),],
            only_extr_columns=['correct_division'],
            reader_cards=make_reader_cards('*_divs.csv'),
            layout_param_group='Divider',
        ),
        # TODO: eliminate this one and move relevant tests to a test not a demo
        SemiDevMeasurementGroup(name='misc_test',
            description='Miscellaneous test measurements',
            meas_columns=[DVColumn('scalar1','int32','Scalar measurement 1')],
            only_extr_columns=[],
            reader_cards=make_reader_cards('NO'), # Not actually reading, just using for other tests
        ),

    )
    trove=ClassicFolderTrove(
        load_info_columns=[DVColumn('Lot', 'string', 'Lot ID of the sample'),
                           DVColumn('Sample', 'string', 'Sample ID of the sample')],
        read_dir=Path(EXAMPLE_DATA_DIR),
    )
        
    return ProjectConfiguration(
        deployment_name='demo1',
        data_definition=SemiDeviceDataDefinition(
            measurement_groups=measurement_groups,
            get_masks_func=lambda: {},
            sample_identifier_column=DVColumn('LotSample','string','Lot and Sample ID of the sample'),
            troves={'':trove},
            layout_params_func=partial(
                get_folder_layout_params,
                layout_params_dir=Path(__file__).parent / 'layout_params',
                layout_params_yaml=Path(__file__).parent / 'layout_params.yaml',),
            sample_info_completer=(lambda d: dict(**d,**(dict(MaskSet='Mask1') if 'MaskSet' not in d else {})))
        ),
        vault=DemoVault(dbname=dbname),
    )



#def complete_matload_info(matload_info:dict):
#    matload_info=matload_info.copy()
#    matload_info['Mask']=matload_info.get('Mask','Mask1')
#    return matload_info