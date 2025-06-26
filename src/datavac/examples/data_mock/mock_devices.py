from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import epsilon_0, Boltzmann, elementary_charge

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
        return VD/self.DCIV(VG=np.array(VG),VD=VD,VS=0,VB=0)['ID']

    def generate_potential_idvg(self,VDs,VGrange, do_plot=False) -> dict:
        # Make a simple IdVg curve
        npoints=51
        VG=np.linspace(VGrange[0], VGrange[1], npoints)
        data={'VG':VG}
        for VD in VDs:
            sweep_data = self.DCIV(VG, VD, VS=0)
            data['fID@VD='+str(VD)]=sweep_data['ID'].to_numpy(dtype='float32')
            data['fIG@VD='+str(VD)]=sweep_data['IG'].to_numpy(dtype='float32')
            data['fIS@VD='+str(VD)]=sweep_data['IS'].to_numpy(dtype='float32') 

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