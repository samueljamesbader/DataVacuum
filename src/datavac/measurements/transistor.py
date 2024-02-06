import dataclasses

import numpy as np
from scipy.signal import savgol_filter

from .measurement_type import MeasurementType
from datavac.util.maths import VTCC
from datavac.util.util import only
from datavac.util.logging import logger


@dataclasses.dataclass
class IdVg(MeasurementType):
    """

    Assumes headers of the form 'VG', 'fID@VD=...', 'fIG@VD=...', etc

    Args:
        norm_column: name of the column to use for calculations requiring normalized current
        Icc: normalized current at which to extract VTcc
        Iswf: swing floor (constant current, normalized, added to abs(I) before extracting SS)
            should be much smaller than the current at which SS is expected to avoid degrading SS,
            but higher than noise floor to ensure noise is not caught as swing!
    """

    norm_column: str
    Icc: float = 1
    Iswf: float = 1e-6
    pol: str = 'n'
    vgoff: float = 0
    abs_vdlin: float = None
    abs_vdsat: float = None

    def __post_init__(self):
        self._norm_col_units={'mm':1e-3,'um':1e-6,'nm':1e-9} \
            [self.norm_column.split("[")[1].split("]")[0]]

    def get_norm(self, measurements):
        return np.array(measurements[self.norm_column],dtype=np.float32)*self._norm_col_units

    def analyze(self, measurements):

        # Properties of the Sav-Gol filter to apply to gm
        gmsavgol=(5,1)
        sssavgol=(3,1)

        # VT CC definition
        logIcc=np.log(self.Icc)

        # Numerical tol to avoid div/0
        tol=1e-14

        VD_strs=[k.split("=")[-1] for k in measurements.headers if k.startswith('fID')]
        if self.abs_vdsat is None:
            VDsat_str=max(VD_strs,key=lambda vds:(-1 if self.pol=='p' else 1)*float(vds))
            VDsat=float(VDsat_str)
        else:
            VDsat=self.abs_vdsat if self.pol=='n' else -self.abs_vdsat
            print(f"Forcing VDsat {VDsat}")
            VDsat_str=next((k for k in VD_strs if np.isclose(float(k),VDsat)),"NOPE")
            if VDsat_str!='NOPE': print(f"VDsat {VDsat} is present among {VD_strs}")
        if self.abs_vdlin is None:
            VDlin_str=min(VD_strs,key=lambda vds:(-1 if self.pol=='p' else 1)*float(vds))
            VDlin=float(VDlin_str)
        else:
            VDlin=self.abs_vdlin if self.pol=='n' else -self.abs_vdlin
            print(f"Forcing VDlin {VDlin}")
            VDlin_str=next((k for k in VD_strs if np.isclose(float(k),VDlin)),"NOPE")
            if VDlin_str!='NOPE': print(f"VDlin {VDlin} is present among {VD_strs}")
        assert VDsat*VDlin>0, "Oops, VDsat and VDlin have different signs"

        W=self.get_norm(measurements)
        VG=measurements['VG']
        IDsat=measurements[f'fID@VD={VDsat_str}'] if f'fID@VD={VDsat_str}' in measurements else VG*np.NaN
        IDlin=measurements[f'fID@VD={VDlin_str}'] if f'fID@VD={VDlin_str}' in measurements else VG*np.NaN
        IGsat=measurements[f'fIG@VD={VDsat_str}'] if f'fIG@VD={VDsat_str}' in measurements else VG*np.NaN

        # Requirements on VG
        assert np.sum(np.abs(np.diff(VG,axis=0)))==0, "Might assume all rows of VG are same for uniform meas"
        VG1d=VG[0,:]
        if self.vgoff is not None:
            ind0=np.argmax(VG1d==self.vgoff)
            if not VG1d[ind0]==self.vgoff:
                logger.warning(f"Must be an exactly {self.vgoff} entry in VG, no tol for this")
                ind0=False
        else: ind0=False
        DVG=VG1d[1]-VG1d[0]
        assert np.allclose(np.diff(VG),DVG), "VG should be even spacing"
        assert np.sign(DVG)==(-1 if self.pol=='p' else 1), "VG should sweep off-to-on"

        gm=savgol_filter(IDsat,*gmsavgol,deriv=1)/DVG
        invswing=savgol_filter(np.log10(np.abs(IDsat.T/W).T+self.Iswf),*sssavgol,deriv=1)/np.abs(DVG)

        all_inds=np.arange(len(VG))
        inds_gmpeak=np.argmax(gm,axis=1)
        gmpeak=gm[all_inds,inds_gmpeak]
        v_gmpeak=VG1d[inds_gmpeak]
        i_gmpeak=abs(IDsat[all_inds,inds_gmpeak])
        vt_gmpeak=v_gmpeak-np.sign(DVG)*i_gmpeak/gmpeak

        measurements['Ion [A]']=np.abs(IDsat[:,-1])
        measurements['Ioff [A]']=measurements['Ion [A]']*np.NaN if (ind0 is False) else np.abs(IDsat[:,ind0])
        measurements['Ioffmin [A]']=np.min(np.abs(IDsat),axis=1)
        measurements['Ioffstart [A]']=np.abs(IDsat[:,0])
        measurements['Ion/Ioff']=measurements['Ion [A]']/measurements['Ioff [A]']
        measurements['Ion/Ioffmin']=measurements['Ion [A]']/measurements['Ioffmin [A]']
        measurements['Ion/Ioffstart']=measurements['Ion [A]']/measurements['Ioffstart [A]']
        measurements['Ron [ohm]']=VDlin/(np.abs(IDlin[:,-1])+tol)
        measurements['VTcc_lin']=VTCC((IDlin.T/W).T,VG,self.Icc,itol=tol)
        measurements['VTcc_sat']=VTCC((IDsat.T/W).T,VG,self.Icc,itol=tol)
        measurements['DIBL']=-(measurements['VTcc_sat']-measurements['VTcc_lin'])/(VDsat-VDlin)
        measurements['VTgm_sat']=vt_gmpeak
        measurements['Gm Peak [S]']=gmpeak
        measurements['SS [mV/dec]']=1e3/np.max(invswing,axis=1)
