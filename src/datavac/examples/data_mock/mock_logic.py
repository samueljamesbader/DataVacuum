from dataclasses import dataclass
from itertools import product
from typing import Union, Optional

import numpy as np
from numpy.fft import irfft
from scipy.fft import rfftfreq, rfft

@dataclass
class InverterDC:
    Vhi: float = 1.0
    """ Logic high voltage """
    Vlo: float = 0.0
    """ Logic low voltage """
    Vim: float = 0.5
    """ Input voltage at which output is halfway between Vhi and Vlo """
    gain: float = 10.0
    """ Max gain (ie negative slope) of the inverter """

    def Vout(self,Vin: np.ndarray[float]):
        Vr=self.Vhi-self.Vlo
        alpha=-self.gain*4
        x= (Vin - self.Vim) / Vr
        return Vr*np.exp(alpha*x)/(1+np.exp(alpha*x)) + self.Vlo
    def generate_potential_iv(self):
        Vin=np.linspace(self.Vlo,self.Vhi,1000)
        Vout=self.Vout(Vin)
        return {'Vin':Vin,f'fVout@VDD={self.Vhi}':Vout,f'rVout@VDD={self.Vhi}':Vout}

    def _ngp(self):
        alpha=-self.gain*4
        Vr=self.Vhi-self.Vlo
        nogainpoint=-Vr/alpha*np.arccosh(-(alpha+2)/2)
        return nogainpoint

    def characteristics(self):
        ngp=self._ngp()
        V_IL= self.Vim - ngp
        V_IH= self.Vim + ngp
        V_OH=self.Vout(V_IL)
        V_OL=self.Vout(V_IH)
        NMH=V_OH-V_IH
        NML=V_IL-V_OL
        return {'V_IL':V_IL,'V_IH':V_IH,'V_OH':V_OH,'V_OL':V_OL,'NMH':NMH,'NML':NML}

    def plot_example(self):
        import matplotlib.pyplot as plt
        curves=self.generate_potential_iv()
        plt.plot(curves['Vin'], curves[f'fVout@VDD={self.Vhi}'])
        plt.axvline(x=self.Vim, ymin=0, ymax=self.Vout(self.Vim), color='k', linestyle='--', label='Vmid')
        ch=self.characteristics()
        plt.axvline(x=ch['V_IH'],ymin=0,ymax=ch['V_OL'], color='r', linestyle='--')
        plt.axhline(y=ch['V_OL'],xmin=ch['V_IH'],xmax=self.Vhi, color='r', linestyle='--')

        plt.axvline(x=ch['V_IL'],ymin=ch['V_OH'],ymax=self.Vhi, color='r', linestyle='--')
        plt.axhline(y=ch['V_OH'],xmin=0,xmax=ch['V_IL'], color='r', linestyle='--')

        #plt.axvline(x=ch['V_IH'],ymin=0,ymax=ch['V_OL'], color='r', linestyle='--')
        plt.xlim(self.Vlo,self.Vhi)
        plt.ylim(self.Vlo,self.Vhi)


def smooth_traces(time:np.ndarray[float], traces:dict[str,np.ndarray[float]],
                  bandwidth:float) -> dict[str,np.ndarray[float]]:
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


@dataclass
class LogicBlock:

    input_names: tuple[str]
    output_names: tuple[str]
    vhi: float = 1
    vlo: float = 0
    bandwidth: float = 1e9

    def truth_table(self,**inputs) -> dict[str,Union[np.ndarray[bool],bool]]:
        raise NotImplementedError

    def vouts(self, time:np.ndarray[float], **inputs) -> dict[str,np.ndarray[float]]:
        vmid=0.5*(self.vhi+self.vlo)
        binned_inputs = {k:(input > vmid) for k,input in inputs.items()}
        binned_outputs= {k:((np.ones_like(time,dtype=bool) & v) if type(v) is bool else v)
                            for k,v in self.truth_table(**binned_inputs).items()}
        ideal_outputs = {k:out*(self.vhi-self.vlo)+self.vlo for k,out in binned_outputs.items()}
        return smooth_traces(time, ideal_outputs, bandwidth=self.bandwidth)

    def generate_potential_digitized_input_sequence(self) -> dict[str,np.ndarray[bool]]:
        digitized_inputs=list(np.array([inp_combo for inp_combo in product([0, 1], repeat=len(self.input_names))]).T)
        return dict(zip(self.input_names,digitized_inputs))

    def generate_potential_inputs(self, clk_period:float, samples_per_period:int, repeats: int, bandwidth: Optional[float])\
                                    -> tuple[np.ndarray[float],dict[np.ndarray[float]]]:
        digitized_inputs=self.generate_potential_digitized_input_sequence()
        if len(digitized_inputs) == 0:
            ideal_inputs={}
            npts=samples_per_period*repeats
        else:
            ideal_inputs={k:np.tile(np.repeat(di,repeats=samples_per_period),repeats)*(self.vhi-self.vlo)+self.vlo
                          for k,di in digitized_inputs.items()}
            npts=len(list(ideal_inputs.values())[0])
        time=np.linspace(0, clk_period*npts, num=npts)
        inputs= smooth_traces(time, ideal_inputs, bandwidth=bandwidth)
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


@dataclass
class TieHi(LogicBlock):
    input_names:tuple[str] = ()
    output_names:tuple[str] = ('o',)
    def truth_table(self) -> dict[str,bool]:
        return {'o': True}


@dataclass
class TieLo(LogicBlock):
    input_names:tuple[str] = ()
    output_names:tuple[str] = ('o',)
    def truth_table(self) -> dict[str,bool]:
        return {'o': False}

@dataclass
class RingOscillator:
    stages: int = 1001
    t_stage: float = 1e-12
    div_by: int = 2**10
    vhi: float = 1.0
    vlo: float = 0.0

    def output_frequency(self):
        return 1/(2*self.t_stage*self.stages) / self.div_by

    def generate_potential_traces(self, n_samples:int = 1000,
                                  sample_rate: float = 10e6, enable_period=60e-6):
        t=np.arange(n_samples)/sample_rate
        f_out=self.output_frequency()
        binary_enable = np.sin(2*np.pi*t/enable_period)<0
        binary_signal = (np.sin(2*np.pi*f_out*t)>0) * binary_enable
        v_out = binary_signal * (self.vhi-self.vlo) + self.vlo
        en_out = binary_enable * (self.vhi-self.vlo) + self.vlo
        return {'Time': t, 'en': en_out, 'out': v_out}

    def plot_example(self):
        import matplotlib.pyplot as plt
        from matplotlib import ticker
        traces=self.generate_potential_traces()
        fig,axes=plt.subplots(len(traces)-1,1,figsize=(10,7))
        for key,ax in zip([t for t in traces if t!='Time'],axes):
            ax.plot(traces['Time'],traces[key],'-o',label=key)
            ax.set_xlabel('Time')
            formatter_x = ticker.EngFormatter(unit="s")
            ax.xaxis.set_major_formatter(formatter_x)
            ax.set_ylabel(key)
            vrange=self.vhi-self.vlo
            ax.set_ylim(self.vlo-.2*vrange,self.vhi+.2*vrange)
        plt.tight_layout()

@dataclass
class Divider:
    div_by: int = 4
    vhi: float = 1.0
    vlo: float = 0.0

    @property
    def vmid(self):
        return 0.5*(self.vhi+self.vlo)

    def generate_potential_traces(self, n_samples:int = 1000,
                                  sample_rate: float = 1e6, input_period=60e-6):
        t=np.arange(n_samples)/sample_rate
        f_in=1/(input_period)
        f_out=f_in/self.div_by
        binary_input  = np.sin(2*np.pi*f_in *t)<0
        binary_output = np.sin(2*np.pi*f_out*t)<0
        v_in  = binary_input  * (self.vhi-self.vlo) + self.vlo
        v_out = binary_output * (self.vhi-self.vlo) + self.vlo
        return {'Time': t, 'a': v_in, 'o': v_out}


    def plot_example(self):
        import matplotlib.pyplot as plt
        traces=self.generate_potential_traces()
        fig,axes=plt.subplots(len(traces)-1,1,figsize=(10,10))
        for key,ax in zip([t for t in traces if t!='Time'],axes):
            ax.plot(traces['Time'],traces[key],label=key)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(key)
            vrange=self.vhi-self.vlo
            ax.set_ylim(self.vlo-.2*vrange,self.vhi+.2*vrange)
        plt.tight_layout()

if __name__=='__main__':
    import matplotlib.pyplot as plt
    InverterDC(Vim=.6, gain=7).plot_example()
    #OrGate().plot_example()
    #DFlipFlop().plot_example()
    #RingOscillator().plot_example()
    #Divider().plot_example()
    plt.show()
