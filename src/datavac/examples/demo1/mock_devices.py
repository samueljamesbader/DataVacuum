from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from numpy.fft import irfft
from pint.testsuite.helpers import requires_babel
from scipy.constants import epsilon_0, Boltzmann, elementary_charge
from scipy.fft import rfftfreq, rfft


# Implements a simple model from http://dx.doi.org/10.1109/ted.2009.2024022
# Plus some arbitrary BTI content and entirely made up noise options
@dataclass
class Transistor4T:
    L: float = 100e-9  # m
    W: float = 1e-6  # m
    mu: float = 100e-4  # m2/Vs
    vx0: float = 1e5  # m/s
    tox: float = 50e-9  # m
    pol: str = 'n'
    VT0: float = .5  # V
    n: float = 1.2
    alpha: float = 0
    delta: float = .01  # V/V
    beta: float = 1.6

    noise_factor_ID: float = 0 # 0.2

    Ea_BTI: float = 1
    beta_BTI: float = 1
    A_BTI: float = .4/(np.exp(1*2.6)*np.exp(-1/(.026))*(10000)**.25)

    def __post_init__(self):
        self._btishift: float = 0

    @property
    def Cox(self):
        return 3.9 * epsilon_0 / self.tox

    def _phiT(self, T=300):
        return Boltzmann * T / elementary_charge

    @property
    def VDSat(self):
        return self.vx0 * self.L / self.mu

    def DCIV(self, VG, VD, VS=None, VB=None, T=300):
        if (nois:=(VS is None)): VS=VD*0
        if (noib:=(VB is None)): VB=VS
        if self.pol == "n":
            VT0 =  self.VT0
        else:
            VG,VD,VS,VB=-VG,-VD,-VS,-VB
            VT0 = -self.VT0

        swapSD=(VD<VS)
        VS,VD=np.choose(swapSD,[VS,VD]),np.choose(swapSD,[VD,VS])

        VGS = VG - VS
        VDS = VD - VS

        phiT = self._phiT(T)
        VT = VT0 - self.delta * VDS + self._btishift
        if self.alpha==0: Ff= 0
        else: Ff = 1 / (1 + np.exp((VGS - (VT - self.alpha * phiT / 2)) / (self.alpha * phiT)))
        VDSat = self.VDSat * (1 - Ff) + phiT * Ff
        Q = self.Cox * self.n * phiT * np.log(1 + np.exp((VGS - (VT - self.alpha * phiT * Ff)) / (self.n * phiT)))
        Fs = (VDS / VDSat) / (1 + (VDS / VDSat) ** self.beta) ** (1 / self.beta)
        ID = self.W * Q * self.vx0 * Fs

        ID *= 1 + np.random.rand(*VG.shape) * self.noise_factor_ID
        IS = -ID
        IG = np.random.rand(*VG.shape) * 1e-13 + 0.00000001 * (20.0 ** (VG - 16))
        IB = (-1 if self.pol=='p' else 1)*np.clip((np.exp((VB-VS)/(.026))-1),0,.1)+np.random.rand(*VG.shape) * 1e-13

        VS,VD=np.choose(swapSD,[VS,VD]),np.choose(swapSD,[VD,VS])
        IS,ID=np.choose(swapSD,[IS,ID]),np.choose(swapSD,[ID,IS])
        if self.pol != "n":
            IG,ID,IB=-IG,-ID,-IB

        data= {'VG':VG,'VD':VD,'ID': ID, 'IS': IS, 'IG': IG, 'IB': IB}
        if nois: del data['IS']
        if noib: del data['IB']
        return pd.DataFrame(data)

    def CV(self, VG, f):
        if self.pol != "n": raise NotImplementedError

        # Assume source and drain connected to lo, and room temp
        VS = 0
        VD = 0
        T = 300

        # Copied from DCIV
        VGS = VG - VS
        VDS = VD - VS

        phiT = self._phiT(T)
        VT = self.VT0 - self.delta * VDS
        Ff = 1 / (1 + np.exp((VGS - (VT - self.alpha * phiT / 2)) / (self.alpha * phiT)))

        # Just ignore the Ff(VGS) dependence..
        # d/dx [ ln(1+e^ax) ] = a / (1+e^-ax)
        dQdV = self.W * self.L * self.Cox \
               / (1 + np.exp(-(VGS - (VT - self.alpha * phiT * Ff)) / (self.n * phiT)))

        G = dQdV * (2 * np.pi * 960e3) / 100
        Cnoise = (np.random.rand(*dQdV.shape) - .5) * .3e-15 * (960e3 / f) ** .5
        return pd.DataFrame({'VG': VG, 'C': dQdV + Cnoise, 'Gp': G})

    def stress(self,duration:float,**voltages):
        VG=voltages['VG']
        assert voltages['VD']==voltages['VS']==voltages['VB']==0
        T=300
        ti_eff=(self._btishift/(self.A_BTI*np.exp(self.beta_BTI*VG)*np.exp(-self.Ea_BTI/self._phiT(T))))**4
        tf_eff=ti_eff+duration
        self._btishift=self.A_BTI*np.exp(self.beta_BTI*VG)*np.exp(-self.Ea_BTI/self._phiT(T))*tf_eff**.25

    def approximate_Ron(self,VG):
        VD=1e-3
        return VD/self.DCIV(VG=np.array([VG]),VD=VD,VS=0,VB=0).iloc[0]['ID']

@dataclass
class LogicBlock:

    input_names: tuple[str]
    output_names: tuple[str]
    vhi: float = 1
    vlo: float = 0
    bandwidth: float = 1e9

    def truth_table(self,**inputs) -> dict[str,np.ndarray[bool]]:
        raise NotImplementedError

    def vouts(self, time:np.ndarray[float], **inputs) -> dict[str,np.ndarray[float]]:
        vmid=0.5*(self.vhi+self.vlo)
        binned_inputs = {k:(input > vmid) for k,input in inputs.items()}
        binned_outputs= self.truth_table(**binned_inputs)
        ideal_outputs = {k:out*(self.vhi-self.vlo)+self.vlo for k,out in binned_outputs.items()}
        return self._smooth_traces(time, ideal_outputs)

    def _smooth_traces(self, time:np.ndarray[float], traces:dict[str,np.ndarray[float]],
                       bandwidth:Optional[float]=None) -> dict[str,np.ndarray[float]]:
        bandwidth=self.bandwidth if bandwidth is None else bandwidth
        dt = time[1] - time[0]
        n = len(time)
        freqs = rfftfreq(n, dt)
        filtered_traces = {}

        for k,trace in traces.items():
            fft_output = rfft(trace)
            fft_output *= 1/(1 + freqs / bandwidth)
            filtered_output = irfft(fft_output).real
            filtered_traces[k]=filtered_output

        return filtered_traces

    def generate_potential_digitized_input_sequence(self) -> dict[str,np.ndarray[bool]]:
        digitized_inputs=list(np.array([inp_combo for inp_combo in product([0, 1], repeat=len(self.input_names))]).T)
        return dict(zip(self.input_names,digitized_inputs))

    def generate_potential_inputs(self, clk_period:float, samples_per_period:int, repeats: int, bandwidth: Optional[float])\
                                    -> tuple[np.ndarray[float],dict[np.ndarray[float]]]:
        digitized_inputs=self.generate_potential_digitized_input_sequence()
        ideal_inputs={k:np.tile(np.repeat(di,repeats=samples_per_period),repeats)*(self.vhi-self.vlo)+self.vlo
                      for k,di in digitized_inputs.items()}
        npts=len(list(ideal_inputs.values())[0])
        time=np.linspace(0, clk_period*npts, num=npts)
        inputs=self._smooth_traces(time, ideal_inputs, bandwidth=bandwidth)
        return time,inputs

    def generate_potential_traces(self, clk_period:float, samples_per_period:int, repeats:int, bandwidth: Optional[float]) \
            -> dict[str,np.ndarray[float]]:
        time,inputs=self.generate_potential_inputs(clk_period,samples_per_period,repeats,bandwidth)
        outputs=self.vouts(time,**inputs)
        return dict(Time=time,**inputs,**outputs)

    def plot_example(self):
        import matplotlib.pyplot as plt
        traces=self.generate_potential_traces(clk_period=1e-8, samples_per_period=20, bandwidth=1e9, repeats=1)
        fig,axes=plt.subplots(len(traces)-1,1,figsize=(10,10))
        for key,ax in zip([t for t in traces if t!='Time'],axes):
            ax.plot(traces['Time'],traces[key],label=key)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(key)
            vrange=self.vhi-self.vlo
            ax.set_ylim(self.vlo-.2*vrange,self.vhi+.2*vrange)
        plt.tight_layout()

@dataclass
class AndGate(LogicBlock):
    input_names:tuple[str] = ('a','b')
    output_names:tuple[str] = ('o',)
    def truth_table(self,a: np.ndarray[bool],b: np.ndarray[bool]) -> dict[str,np.ndarray[bool]]:
        return {'o':a & b}

@dataclass
class OrGate(LogicBlock):
    input_names:tuple[str] = ('a','b')
    output_names:tuple[str] = ('o',)
    def truth_table(self,a: np.ndarray[bool],b: np.ndarray[bool]) -> dict[str,np.ndarray[bool]]:
        return {'o':a | b}


@dataclass
class DFlipFlop(LogicBlock):
    input_names:tuple[str] = ('rb','d','clk')
    output_names:tuple[str] = ('o',)
    def truth_table(self, clk: np.ndarray[bool], d: np.ndarray[bool], rb: np.ndarray[bool]) -> dict[str,np.ndarray[bool]]:
        o_n=[False]
        for i in range(1,len(clk)):
            o_nm1 = o_n[i-1]
            clk_n = clk[i]
            clk_nm1 = clk[i-1]
            d_n = d[i]
            d_nm1 = d[i-1]
            rb_n = rb[i]
            rb_nm1 = rb[i-1]
            o_n.append(rb_n & ((d_nm1 & (clk_n & ~ clk_nm1)) | (o_nm1 & ~ (clk_n & ~ clk_nm1))))
        return {'o': np.array(o_n,dtype=bool)}

    def generate_potential_digitized_input_sequence(self) -> dict[str,np.ndarray[bool]]:
        return {'rb' :np.array([0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],dtype=bool),
                'd'  :np.array([0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,0],dtype=bool),
                'clk':np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],dtype=bool)
                }

if __name__=='__main__':
    import matplotlib.pyplot as plt
    #OrGate().plot_example()
    DFlipFlop().plot_example()
    plt.show()