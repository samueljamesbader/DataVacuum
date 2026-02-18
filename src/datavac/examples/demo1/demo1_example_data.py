import shutil
import numpy as np

from datavac.examples.example_data import write_example_data_file, get_transistor, get_capacitor, get_inverter, get_ring
from datavac.examples.data_mock.mock_logic import AndGate, OrGate, DFlipFlop, TieHi, TieLo, Divider



def make_example_data_demo1():
    from datavac.examples.demo1 import EXAMPLE_DATA_DIR

    # Clear the directory if it exists and remake it
    if EXAMPLE_DATA_DIR.exists(): shutil.rmtree(EXAMPLE_DATA_DIR)
    EXAMPLE_DATA_DIR.mkdir(parents=True,exist_ok=True)

    ###
    # nMOS Id-Vgs
    ###
    data={structure: get_transistor('Mask1',structure).generate_potential_idvg(VDs=[ .01,  1], VGrange=[0,  1], do_plot=False)
          for structure in ['nmos1','nmos2','nmos3']}
    write_example_data_file('lot1','sample1','nMOS_IdVg',data, EXAMPLE_DATA_DIR)

    ###
    # pMOS Id-Vgs
    ###
    data={structure: get_transistor('Mask1',structure).generate_potential_idvg(VDs=[-.01, -1], VGrange=[0, -1], do_plot=False)
          for structure in ['pmos1','pmos2','pmos3']}
    write_example_data_file('lot1','sample1','pMOS_IdVg',data, EXAMPLE_DATA_DIR)

    ###
    # Capacitor CVs
    ###
    data={structure: get_capacitor('Mask1',structure).generate_potential_cv(VArange=(-1,1),freqs=['1k','10k'], do_plot=False)
          for structure in ['cap1','cap2']}
    write_example_data_file('lot1','sample1','Cap_CV',data, EXAMPLE_DATA_DIR)
    data={structure: get_capacitor('Mask1',structure).generate_potential_cv(VArange=(-1,1),freqs=['1k','10k'], do_plot=False)
          for structure in ['op1','op2']}
    write_example_data_file('lot1','sample1','Open_CV',data, EXAMPLE_DATA_DIR)

    ###
    # DC inverters
    ###
    data={structure: get_inverter('Mask1',structure).generate_potential_iv() for structure in ['inv1','inv2']}
    write_example_data_file('lot1','sample1','invs',data, EXAMPLE_DATA_DIR)

    ###
    # Tie-Hi
    ###
    data={
        'good_tihi': TieHi().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=1, bandwidth=1e9),
        'bad_tihi': TieLo().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=1, bandwidth=1e9),
    }
    write_example_data_file('lot1','sample1','tiehi_logic',data, EXAMPLE_DATA_DIR)

    ###
    # OR gates
    ###
    data={
        'good_or': OrGate().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=2, bandwidth=1e9),
        'bad_or': AndGate().generate_potential_traces(clk_period=1e-9, samples_per_period=20, repeats=2, bandwidth=1e9),
    }
    write_example_data_file('lot1','sample1','orcell_logic',data, EXAMPLE_DATA_DIR)

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
    write_example_data_file('lot1','sample1','dff_logic',data, EXAMPLE_DATA_DIR)

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
    write_example_data_file('lot1','sample1','divs',data, EXAMPLE_DATA_DIR)

    ###
    # ROs
    ###
    data={structure: get_ring('Mask1',structure).generate_potential_traces(n_samples=5000*3,sample_rate=10e6,enable_period=500e-6,) for structure in ['ro1','ro2','ro3','ro4']}
    write_example_data_file('lot1','sample1','ros',data, EXAMPLE_DATA_DIR)


if __name__ == '__main__':
    make_example_data_demo1()