#def make_KelvinRon(transistor:Transistor4T,IDs,VGrange, do_plot=False) -> dict:
#    # Make a simple IdVg curve
#    npoints=51
#    VG=np.linspace(VGrange[0], VGrange[1], npoints)
#    data={'VG':VG}
#    try_ron = transistor.approximate_Ron(VG)
#    for ID in IDs:
#        try_vd=try_ron*ID
#        sweep_data= transistor.DCIV(VG, try_vd, VS=0)
#        data[f'fVDS@ID={ID}']


from datavac.config.layout_params import LP
from datavac.examples.data_mock.mock_devices import Capacitor, Transistor4T
from datavac.examples.data_mock.mock_logic import InverterDC, RingOscillator
import numpy as np
import pandas as pd


from pathlib import Path
from typing import Mapping, Optional


def write_example_data_file(lot,sample,meas_name,data_dicts:Mapping[str,Mapping[str,np.ndarray]],EXAMPLE_DATA_DIR:Path, subdir=""):
    data=pd.concat([pd.DataFrame(subdata).assign(**{'Site':site,'MeasNo':i}) for i,(site,subdata) in enumerate(data_dicts.items())])
    (EXAMPLE_DATA_DIR/lot/subdir).mkdir(parents=True,exist_ok=True)
    data.to_csv(EXAMPLE_DATA_DIR/lot/subdir/f"{lot}_{sample}_{meas_name}.csv",index=False)


def read_csv(file:Path, mg_name: str, only_sampleload_info: dict = {},
             read_info_so_far: Optional[dict] = None) -> list[pd.DataFrame]:
    assert len(only_sampleload_info) == 0, "Only sampleload_info is not supported in this example"
    rawcsv=pd.read_csv(file)
    data=[{'RawData':{k:np.array(v,dtype=np.float32) for k,v in grp.drop(columns=['Site','MeasNo']).to_dict('list').items()},
           'Site':grp['Site'].iloc[0],
           'MeasLength':len(grp),}
             for csv,grp in rawcsv.groupby('MeasNo')]
    return [pd.DataFrame(data).convert_dtypes()]


def get_transistor(mask,structure):
    lp=LP()
    site_params=lp.get_params([structure],mask=mask).iloc[0]
    return Transistor4T(W=site_params['W [um]']*1e-6,L=site_params['L [um]']*1e-6,
               VT0=site_params['VT0_target [V]'],n=site_params['n_target'],pol=site_params['pol'])


def get_capacitor(mask,structure):
    lp=LP()
    site_params=lp.get_params([structure],mask=mask).iloc[0]
    return Capacitor(capacitance=site_params['C_target']+site_params['C_route'])


def get_inverter(mask,structure):
    lp=LP()
    site_params=lp.get_params([structure],mask=mask).iloc[0]
    return InverterDC(Vim=site_params['target_Vmid [V]'], gain=site_params['target_gain'])


def get_ring(mask,structure) -> RingOscillator:
    lp=LP()
    site_params=lp.get_params([structure],mask=mask).iloc[0]
    return RingOscillator(stages=site_params['stages'],
                          t_stage=site_params['target_t_stage [ps]']/1e12,
                          div_by=site_params['div_by'])