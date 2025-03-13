import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import platformdirs

from datavac.examples.demo1 import READ_DIR
from datavac.examples.demo1.mock_devices import Transistor4T
from datavac.io.layout_params import get_layout_params


def make_IdVg(transistor,VDs,VGrange, do_plot=False) -> dict:
    # Make a simple IdVg curve
    npoints=51
    VG=np.linspace(VGrange[0], VGrange[1], npoints)
    data={'VG':VG}
    for VD in VDs:
        sweep_data = transistor.DCIV(VG, VD, VS=0)
        data['fID@VD='+str(VD)]=sweep_data['ID']
        data['fIG@VD='+str(VD)]=sweep_data['IG']
        data['fIS@VD='+str(VD)]=sweep_data['IS']

    if do_plot:
        # Plotting the IdVg data
        import matplotlib.pyplot as plt

        plt.figure()
        for VD in VDs:
            plt.plot(data['VG'], data[f'fID@VD={VD}'], label=f'VD={VD}V')

        plt.xlabel('VG (V)')
        plt.ylabel('ID (A)')
        plt.title('Id-Vg Transfer Curve')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.show()

    return data

def make_KelvinRon(transistor,IDs,VGrange, do_plot=False) -> dict:
    # Make a simple IdVg curve
    npoints=51
    VG=np.linspace(VGrange[0], VGrange[1], npoints)
    data={'VG':VG}
    for ID in IDs:
        sweep_data = transistor.DCIV(VG, VD, VS=0)

def write_example_data_file(lot,sample,meas_name,data_dicts:dict[str,dict[str,np.ndarray]]):
    data=pd.concat([pd.DataFrame(subdata).assign(**{'Site':site,'MeasNo':i}) for i,(site,subdata) in enumerate(data_dicts.items())])
    (READ_DIR/lot).mkdir(parents=True,exist_ok=True)
    data.to_csv(READ_DIR/lot/f"{lot}_{sample}_{meas_name}.csv",index=False)

def read_csv(file,meas_type,meas_group,only_matload_info=None):
    rawcsv=pd.read_csv(READ_DIR/file)
    data=[{'RawData':grp.drop(columns=['Site','MeasNo']).to_dict('list'),
           'Site':grp['Site'].iloc[0],
           'MeasLength':len(grp),}
             for csv,grp in rawcsv.groupby('MeasNo')]
    return [pd.DataFrame(data).convert_dtypes()]

def get_transistor(mask,structure):
    lp=get_layout_params()
    site_params=lp.get_params([structure],mask=mask).iloc[0]
    return Transistor4T(W=site_params['W [um]']*1e-6,L=site_params['L [um]']*1e-6,
               VT0=site_params['VT0_target [V]'],n=site_params['n_target'],pol=site_params['pol'])

def make_example_data():

    # Clear the directory if it exists and remake it
    if READ_DIR.exists(): shutil.rmtree(READ_DIR)
    READ_DIR.mkdir(parents=True,exist_ok=True)

    data={structure: make_IdVg(get_transistor('Mask1',structure), VDs=[ .01,  1], VGrange=[0,  1], do_plot=False)
          for structure in ['nmos1','nmos2','nmos3']}
    write_example_data_file('lot1','sample1','nMOS_IdVg',data)
    data={structure: make_IdVg(get_transistor('Mask1',structure), VDs=[-.01, -1], VGrange=[0, -1], do_plot=False)
          for structure in ['pmos1','pmos2','pmos3']}
    write_example_data_file('lot1','sample1','pMOS_IdVg',data)



if __name__ == '__main__':
    make_example_data()