import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import platformdirs

from datavac.examples.demo1 import READ_DIR
from datavac.examples.demo1.mock_devices import Transistor4T
from datavac.examples.demo1.mock_logic import AndGate, OrGate, DFlipFlop, TieHi, TieLo, RingOscillator, Divider, \
    InverterDC
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

def get_inverter(mask,structure):
    lp=get_layout_params()
    site_params=lp.get_params([structure],mask=mask).iloc[0]
    return InverterDC(Vmid=site_params['target_Vmid [V]'],gain=site_params['target_gain'])

def get_ring(mask,structure) -> RingOscillator:
    lp=get_layout_params()
    site_params=lp.get_params([structure],mask=mask).iloc[0]
    return RingOscillator(stages=site_params['stages'],
                          t_stage=site_params['target_t_stage [ps]']/1e12,
                          div_by=site_params['div_by'])


def make_example_data():

    # Clear the directory if it exists and remake it
    if READ_DIR.exists(): shutil.rmtree(READ_DIR)
    READ_DIR.mkdir(parents=True,exist_ok=True)

    ###
    # nMOS Id-Vgs
    ###
    data={structure: make_IdVg(get_transistor('Mask1',structure), VDs=[ .01,  1], VGrange=[0,  1], do_plot=False)
          for structure in ['nmos1','nmos2','nmos3']}
    write_example_data_file('lot1','sample1','nMOS_IdVg',data)

    ###
    # pMOS Id-Vgs
    ###
    data={structure: make_IdVg(get_transistor('Mask1',structure), VDs=[-.01, -1], VGrange=[0, -1], do_plot=False)
          for structure in ['pmos1','pmos2','pmos3']}
    write_example_data_file('lot1','sample1','pMOS_IdVg',data)

    ###
    # DC inverters
    ###
    data={structure: get_inverter('Mask1',structure).generate_potential_iv() for structure in ['inv1','inv2']}
    write_example_data_file('lot1','sample1','invs',data)

    ###
    # Tie-Hi
    ###
    data={
        'good_tihi': TieHi().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=1, bandwidth=1e9),
        'bad_tihi': TieLo().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=1, bandwidth=1e9),
    }
    write_example_data_file('lot1','sample1','tiehi_logic',data)

    ###
    # OR gates
    ###
    data={
        'good_or': OrGate().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=2, bandwidth=1e9),
        'bad_or': AndGate().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=2, bandwidth=1e9),
    }
    write_example_data_file('lot1','sample1','orcell_logic',data)

    ###
    # D Flip-Flops
    ###
    spp=20
    data={
        'good_dff1': DFlipFlop().generate_potential_traces(clk_period=1e-9, samples_per_period=spp, repeats=2, bandwidth=1e9),
        'bad_dff':   DFlipFlop().generate_potential_traces(clk_period=1e-9, samples_per_period=spp, repeats=2, bandwidth=1e9),
        'good_dff2': DFlipFlop().generate_potential_traces(clk_period=1e-9, samples_per_period=spp, repeats=2, bandwidth=1e9),
    }
    # Flip whole output for bad D Flip-Flop
    data['bad_dff']['o'] = 1-data['bad_dff']['o']
    # Flip just first period, shouldn't matter because output is undefined
    data['good_dff2']['o'][:spp] = 1-data['good_dff2']['o'][:spp]
    write_example_data_file('lot1','sample1','dff_logic',data)

    ###
    # Divs
    ###
    data={}
    data['good_div2']=Divider(div_by=2).generate_potential_traces()
    data['bad_div2']= Divider(div_by=3).generate_potential_traces()
    data['good_div4']=Divider(div_by=4).generate_potential_traces()
    data['bad_div4']= Divider(div_by=3).generate_potential_traces()
    bd4a=data['bad_div4']['a']
    bd4o=data['bad_div4']['o']
    iglitch_start=np.arange(len(bd4a)-1)[np.diff(bd4a>Divider().vmid)!=0][5]+1
    iglitch_stop= np.arange(len(bd4a)-1)[np.diff(bd4a>Divider().vmid)!=0][6]+1
    bd4o[iglitch_start:iglitch_stop]=Divider().vhi-bd4o[iglitch_start:iglitch_stop]
    write_example_data_file('lot1','sample1','divs',data)

    ###
    # ROs
    ###
    data={structure: get_ring('Mask1',structure).generate_potential_traces(n_samples=5000*3,sample_rate=10e6,enable_period=500e-6,) for structure in ['ro1','ro2','ro3','ro4']}
    write_example_data_file('lot1','sample1','ros',data)


if __name__ == '__main__':
    make_example_data()