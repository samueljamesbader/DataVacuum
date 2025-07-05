import ast
from dataclasses import dataclass
from typing import Sequence, Optional

from datavac.config.data_definition import DVColumn
from datavac.measurements.measurement_group import SemiDevMeasurementGroup
from datavac.util.util import asnamedict
import numpy as np
import pandas as pd
from scipy.fft import rfftfreq, rfft

from datavac.io.measurement_table import UniformMeasurementTable
from datavac.util.maths import YatX


@dataclass(eq=False, repr=False)
class InverterDC(SemiDevMeasurementGroup):
    vhi: float = 1.0
    vlo: float = 0.0
    def extract_by_umt(self, measurements: 'UniformMeasurementTable'):
        assert self.vlo==0
        Vim=YatX(X=measurements[f'fVout@VDD={self.vhi}'],Y=measurements['Vin'],
                  x=(self.vhi+self.vlo)/2,reverse_crossing=True)
        measurements['Vim [V]']=Vim
        Vm=YatX(X=measurements[f'fVout@VDD={self.vhi}']-measurements['Vin'],
                Y=(measurements[f'fVout@VDD={self.vhi}']+measurements['Vin'])/2,
                x=0,reverse_crossing=True)
        measurements['Vm [V]']=Vm

        dVin=np.mean(np.diff(measurements['Vin'],axis=1))
        gain=np.diff(measurements[f'fVout@VDD={self.vhi}'],axis=1)/dVin
        Vinm=(measurements['Vin'][:,:-1]+measurements['Vin'][:,1:])/2
        Voutm=(measurements[f'fVout@VDD={self.vhi}'][:,:-1]+measurements[f'fVout@VDD={self.vhi}'][:,1:])/2
        measurements['max_gain']=np.max(-gain,axis=1)
        after_vmid=(Vinm.T>Vm).T
        nan_after_vmid=np.ones_like(after_vmid,dtype=float); nan_after_vmid[ after_vmid]=np.nan
        nan_befor_vmid=np.ones_like(after_vmid,dtype=float); nan_befor_vmid[~after_vmid]=np.nan

        gainlef=gain*nan_after_vmid
        gainrit=gain*nan_befor_vmid
        # https://web.mit.edu/6.012/www/SP07-L11.pdf
        measurements['VIL [V]']=YatX(Y=Vinm,X=gainlef,x=-1,reverse_crossing=True)
        measurements['VIH [V]']=YatX(Y=Vinm,X=gainrit,x=-1,reverse_crossing=False)
        measurements['VOL [V]']=YatX(Y=Voutm,X=gainrit,x=-1,reverse_crossing=False)
        measurements['VOH [V]']=YatX(Y=Voutm,X=gainlef,x=-1,reverse_crossing=True)
        measurements['NMH [V]']=measurements['VOH [V]']-measurements['VIH [V]']
        measurements['NML [V]']=measurements['VIL [V]']-measurements['VOL [V]']

    def available_extr_columns(self) -> dict[str, DVColumn]:
        return asnamedict(
            DVColumn('VIL [V]', 'float', 'Inverter DC VIL'),
            DVColumn('VIH [V]', 'float', 'Inverter DC VIH'),
            DVColumn('VOL [V]', 'float', 'Inverter DC VOL'),
            DVColumn('VOH [V]', 'float', 'Inverter DC VOH'),
            DVColumn('NML [V]', 'float', 'Inverter DC NML'),
            DVColumn('NMH [V]', 'float', 'Inverter DC NMH'),
            DVColumn('max_gain', 'float', 'Maximum gain of the inverter')
        )


def get_digitized_curves(time:np.ndarray[float], curves:dict[str,np.ndarray[float]], vmid: float,
                         tcluster_thresh: Optional[float] = None) -> dict[str,np.ndarray[bool]]:

    # Note curves should be the k.split("_")[0] of the other curves dict

    # sample period
    tstep=np.mean(np.diff(time,axis=1))


    raw_bin_for_clusters= {k: c>vmid for k, c in curves.items()}
    arng=np.vstack([np.arange(time.shape[1]-1) for i in range(time.shape[0])])
    #import pdb; pdb.set_trace()
    iflips=sorted([i for k in raw_bin_for_clusters for i in arng[np.diff(raw_bin_for_clusters[k])!=0]])
    iflips = [0,*iflips,time.shape[1]-1]

    # If a clustering threshold is specified in time, translate it to points
    if tcluster_thresh is not None:
        iclusterthresh = int(round(tcluster_thresh/tstep))
    # Otherwise calculate a good threshold as a fraction of the max interval between any flips
    else:
        iclusterthresh=int(round(np.max(np.diff(iflips))/4))

    # Cluster the individual transitions within the clustering threshold to find a discrete transition set
    clusters=[[iflips[0]]]
    for i in iflips:
        if i-clusters[-1][-1] > iclusterthresh: clusters.append([i])
        else: clusters[-1].append(i)
    iflips=[int(round(np.median(c))) for c in clusters]

    # Use those transition times to break up the data and average truth values over time within inter-transition
    digitized_curves={k:np.vstack([np.mean(curves[k][:,imin:imax],axis=1)>vmid
                                   for imin,imax in zip(iflips[:-1],iflips[1:])]).T
                      for k in curves}
    return digitized_curves

@dataclass(eq=False, repr=False)
class OscopeFormulaLogic(SemiDevMeasurementGroup):
    formula_col: str = 'formula'
    time_col: str = 'Time'
    vhi: float = 1
    vlo: float = 0
    channel_mapping:Optional[dict[str,Sequence[str]]]=None
    tcluster_thresh: float = None

    def extract_by_umt(self, measurements: UniformMeasurementTable):

        # Will be populated as we go with correctness evaluations
        truth_table_pass=pd.Series([pd.NA]*len(measurements),dtype='boolean')

        for formula, grp in measurements.scalar_table_with_layout_params([self.formula_col]).groupby(self.formula_col):
            # Hopefully temporary, but since "|" is used as a delimiter for CSV uploading,
            # it can't appear in the formula in the layout params table
            formula:str=formula.replace("%OR%","|")

            # Parse the formula to the names of inputs and outputs
            outside,inpside=formula.split("=")
            tree = ast.parse(inpside.strip())
            inp_names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
            inp_names = [n for n in inp_names if n not in ['TRUE','FALSE']]
            out_names = [on.strip() for on in outside.split(",")]

            # Create a dictionary of curves needed to evaluate the formula's correctness
            channel_mapping={h:[h] for h in measurements.headers} if self.channel_mapping is None else self.channel_mapping
            curves={name:measurements[ch][list(grp.index),:] for ch, names in channel_mapping.items()
                    for name in names if ch in measurements.headers
                                            and name in [n.split("_")[0] for n in inp_names+out_names]}

            digitized_curves=get_digitized_curves(measurements[self.time_col][list(grp.index)],
                                                  {k.split("_")[0]:c for k,c in curves.items()},
                                                  vmid=(self.vhi+self.vlo)/2, tcluster_thresh=self.tcluster_thresh)

            nd=len(list(digitized_curves.values())[0])

            # Form a dictionary of inputs to the formula
            # If the formula involves a reference to previous (n-1) state, then don't care about state at first
            inps = {}
            first_nocare=False
            for k in inp_names:
                if k.endswith("_n"):
                    inps[k] = digitized_curves[k.split("_")[0]]
                elif k.endswith("_nm1"):
                    inps[k] = np.roll(digitized_curves[k.split("_")[0]],shift=1,axis=1)
                    first_nocare=True
                else:
                    raise Exception(f"Input '{k}' should end with _n or _nm1")
            inps['TRUE'] =  np.ones([len(grp),nd],dtype=bool)
            inps['FALSE'] = np.zeros([len(grp),nd],dtype=bool)

            # Compute the expected output
            calc_outs = dict(zip([k.strip() for k in outside.split(",")],eval(inpside.strip(),{"__builtins__": None}, inps)))

            # Compare to measured
            ttfails=[]
            for k in calc_outs:
                mask=slice(None) if not first_nocare else slice(1,None)
                assert k.endswith("_n"), f"How is there an output curve '{k}' that doesn't end with _n?"
                ttptfail= digitized_curves[k.split("_n")[0]][:,mask]!=calc_outs[k][mask]
                ttfails.append(np.sum(ttptfail,axis=1))
            truth_table_pass[grp.index]=(np.sum(ttfails,axis=0)==0)

        measurements['truth_table_pass']=truth_table_pass
    def available_extr_columns(self) -> dict[str, DVColumn]:
        return asnamedict(
            DVColumn('truth_table_pass', 'boolean', 'Whether measured curves satisfy the truth table formula')
        )


@dataclass(eq=False, repr=False)
class OscopeRingOscillator(SemiDevMeasurementGroup):
    time_col: str = 'Time'
    signal_col: str = 'out'
    enable_col: Optional[str] = 'en'
    div_by_col: Optional[str] = None
    stages_col: Optional[str] = None
    vhi: float = 1.0
    vlo: float = 0.0

    @property
    def vmid(self): return (self.vhi+self.vlo)/2

    def get_div_by(self, measurements: UniformMeasurementTable):
        return measurements.scalar_table_with_layout_params([self.div_by_col])[self.div_by_col]\
            if self.div_by_col else 1

    def extract_by_umt(self, measurements: UniformMeasurementTable):
        time: np.ndarray[float]=measurements[self.time_col]
        signal: np.ndarray[float]=measurements[self.signal_col]
        enable: np.ndarray[float]=measurements[self.enable_col]\
            if self.enable_col else np.ones_like(signal, dtype=bool)

        # Find the first block where enable is true from all measurements
        en1d = np.all(enable>self.vmid,axis=0)
        assert np.any(en1d), f"No all-enabled blocks found in {self.enable_col} column"
        en1d_start = np.argmax(en1d)
        en1d_stop = np.argmax((~en1d)*(np.arange(len(en1d))>=en1d_start))
        if en1d_stop==en1d_start: en1d_stop=-1

        # Signal within first enabled block
        val_signal = signal[:,en1d_start:en1d_stop]

        # FFT within first enabled block
        N = val_signal.shape[1]
        dt = np.mean(time[:,-1]-time[:,0])/(time.shape[-1]-1)
        fft_vals=rfft((val_signal.T - np.mean(val_signal,axis=1)).T)
        freqs=rfftfreq(N,dt)

        # Get frequency of signal
        measurements['f_out_fft [Hz]']=freqs[np.argmax(np.abs(fft_vals),axis=1)]
        measurements['f_out_fine [Hz]']=fine_freq_of_signal(val_signal>self.vmid,
                                            dt=dt, expected_interval=1/(2*measurements['f_out_fft [Hz]']*dt))

        # If the signal is known to be divided, multiply back up
        measurements['f_osc [Hz]']=measurements['f_out_fine [Hz]']*self.get_div_by(measurements)
        #assert np.all(np.isfinite(measurements['f_osc [Hz]']))

        # Normalize by stages
        if self.stages_col:
            stages=measurements.scalar_table_with_layout_params([self.stages_col])[self.stages_col]
            measurements['t_stage [s]']=1/(2*stages*measurements['f_osc [Hz]'])
            measurements['t_stage [ps]']=measurements['t_stage [s]']*1e12
    def available_extr_columns(self) -> dict[str, DVColumn]:
        return asnamedict(
            DVColumn('f_out_fft [Hz]', 'float', 'DUT output signal frequency computed by FFT'),
            DVColumn('f_out_fine [Hz]', 'float', 'DUT output signal frequency refined by counting transitions, if clean'),
            DVColumn('f_osc [Hz]', 'float', 'Oscillator frequency (f_out_fine scaled by the divider ratio to get ring oscillator itself)'),
            DVColumn('t_stage [s]', 'float', 'Delay time per stage (in seconds), based on f_osc'),
            DVColumn('t_stage [ps]', 'float', 'Delay time per stage (in picoseconds), based on f_osc')
        )

class OscopeDivider(SemiDevMeasurementGroup):
    time_col: str = 'Time'
    input_col: str = 'a'
    output_col: str = 'o'
    div_by_col: str = 'div_by'
    vhi: float = 1.0
    vlo: float = 0.0

    def extract_by_umt(self, measurements: 'UniformMeasurementTable'):
        correct_division=pd.Series([pd.NA]*len(measurements),dtype='boolean')
        internal_phase  =pd.Series([-1]*len(measurements),dtype='int32')

        for divby, grp in measurements.scalar_table_with_layout_params([self.div_by_col],on_missing='ignore').groupby(self.div_by_col):

            digcs=get_digitized_curves(measurements[self.time_col][grp.index],
                                       {'in':measurements[self.input_col][grp.index],
                                        'out':measurements[self.output_col][grp.index]},
                                       vmid=(self.vhi+self.vlo)/2)

            # Compute a 3-D array, where the first index runs over possible "initial settings" of the counter
            # and each 2-D array is a potentially correct output for that initial setting
            transition_count = np.pad(np.cumsum(np.diff(digcs['in'],axis=1)!=0,axis=1),
                                      ((0, 0), (1, 0)), mode='constant', constant_values=0)
            potential_correct_outputs=np.array([
                (np.mod(transition_count+sh, 2*divby)<divby)
                for sh in range(2*divby)
            ])

            # Check if each row of output matches at least one computed output
            matches=np.all(digcs['out']==potential_correct_outputs,axis=2)
            correct_ingrp=np.any(matches,axis=0)

            # Moreover, figure out which match that is
            iphase_ingrp=np.asarray(np.argmax(matches,axis=0),dtype='int32')
            iphase_ingrp[~correct_ingrp]=-1

            # Assign with appropriate indices
            correct_division[grp.index]=correct_ingrp
            internal_phase[grp.index]=iphase_ingrp

        measurements['correct_division']=correct_division
        measurements['internal_phase']=internal_phase

    def available_extr_columns(self) -> dict[str, DVColumn]:
        return asnamedict(
            DVColumn('correct_division', 'boolean', 'Whether the divider output is correct'),
            DVColumn('internal_phase', 'int32', 'Internal phase of the divider, -1 if not correct')
        )

def fft_freq_of_sig(signal: np.ndarray[float], dt: float) -> np.ndarray[float]:
    N = signal.shape[1]
    fft_vals=rfft((signal.T-np.mean(signal,axis=1)).T)
    freqs=rfftfreq(N,dt)
    return freqs[np.argmax(np.abs(fft_vals),axis=1)]

def fine_freq_of_signal(binary_signal:np.ndarray[bool], dt:float, expected_interval:float):
    arng=np.arange(binary_signal.shape[1]-1)
    flipmat=(np.diff(binary_signal,axis=1)!=0)
    glitch_free=[]
    fine_freq=[]
    for row,flps in enumerate(flipmat):
        iflips=arng[flps]
        if len(iflips)<2:
            glitch_free.append(False)
            fine_freq.append(np.nan)
            continue
        try:
            if np.allclose(np.diff(iflips),expected_interval[row], rtol=.1, atol=2.5) \
                    and iflips[0]<expected_interval[row]*1.2 \
                    and iflips[-1]>binary_signal.shape[1]-expected_interval[row]*1.2:
                glitch_free.append(True)
                #md=np.median(np.diff(iflips))
                md=np.mean(np.diff(iflips))
                assert np.isclose(md,expected_interval[row],rtol=.2), f"Unexpectedly large interval refinement {md} vs {expected_interval[row]}"
                fine_freq.append(1/(md*dt*2))
            else:
                print(f"glitch at {row} with iflips={iflips}, expected_interval={expected_interval[row]}")
                glitch_free.append(False)
                fine_freq.append(np.nan)
        except IndexError:
            glitch_free.append(False)
            fine_freq.append(np.nan)
    return np.array(fine_freq,dtype='float64')
