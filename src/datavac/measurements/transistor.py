import dataclasses
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .measurement_type import MeasurementType
from datavac.util.maths import VTCC, YatX
from datavac.util.util import only
from datavac.util.logging import logger
from ..io.measurement_table import MeasurementTable

@dataclasses.dataclass
class MeasurementWithLinearNormColumn(MeasurementType):
    norm_column: str = None
    def __post_init__(self):
        if self.norm_column is not None:
            self._norm_col_units={'mm':1e-3,'um':1e-6,'nm':1e-9} \
                [self.norm_column.split("[")[1].split("]")[0]]

    def get_norm(self, measurements: MeasurementTable):
        if self.norm_column is None:
            return None
        return np.array(measurements \
                        .scalar_table_with_layout_params(params=[self.norm_column],on_missing='ignore')[self.norm_column],dtype=np.float32) \
            *self._norm_col_units

@dataclasses.dataclass
class IdVg(MeasurementWithLinearNormColumn):
    """

    Assumes headers of the form 'VG', 'fID@VD=...', 'fIG@VD=...', etc

    Args:
        norm_column: name of the column to use for calculations requiring normalized current
        Iccs: normalized currents at which to extract VTcc
        Iswf: swing floor (constant current, normalized, added to abs(I) before extracting SS)
            should be much smaller than the current at which SS is expected to avoid degrading SS,
            but higher than noise floor to ensure noise is not caught as swing!
    """

    Iccs: dict[str,float] = dataclasses.field(default_factory=lambda:{'':1})
    Iswf: float = 1e-6
    pol: str = 'n'
    vgoff: float = 0
    Vgons: dict[str,float] = dataclasses.field(default_factory=lambda:{'':1})
    abs_vdlin: float = None
    abs_vdsat: float = None


    def analyze(self, measurements):

        # Properties of the Sav-Gol filter to apply to gm
        gmsavgol=(5,1)
        sssavgol=(3,1)

        # Numerical tol to avoid div/0
        tol=1e-14

        VD_strs=[k.split("=")[-1] for k in measurements.headers if k.startswith('fID')]
        if self.abs_vdsat is None:
            VDsat_str=max(VD_strs,key=lambda vds:(-1 if self.pol=='p' else 1)*float(vds))
            VDsat=float(VDsat_str)
        else:
            VDsat=self.abs_vdsat if self.pol=='n' else -self.abs_vdsat
            #print(f"Forcing VDsat {VDsat}")
            VDsat_str=next((k for k in VD_strs if np.isclose(float(k),VDsat)),"NOPE")
            #if VDsat_str!='NOPE': print(f"VDsat {VDsat} is present among {VD_strs}")
        if self.abs_vdlin is None:
            VDlin_str=min(VD_strs,key=lambda vds:(-1 if self.pol=='p' else 1)*float(vds))
            VDlin=float(VDlin_str)
        else:
            VDlin=self.abs_vdlin if self.pol=='n' else -self.abs_vdlin
            #print(f"Forcing VDlin {VDlin}")
            VDlin_str=next((k for k in VD_strs if np.isclose(float(k),VDlin)),"NOPE")
            #if VDlin_str!='NOPE': print(f"VDlin {VDlin} is present among {VD_strs}")
        assert VDsat*VDlin>0, "Oops, VDsat and VDlin have different signs"

        W=self.get_norm(measurements)
        VG=measurements['VG']
        IDsat=measurements[f'fID@VD={VDsat_str}'] if (has_idsat:=(f'fID@VD={VDsat_str}' in measurements)) else VG*np.NaN
        IDlin=measurements[f'fID@VD={VDlin_str}'] if (has_idlin:=(f'fID@VD={VDlin_str}' in measurements)) else VG*np.NaN
        IGsat=measurements[f'fIG@VD={VDsat_str}'] if (has_igsat:=(f'fIG@VD={VDsat_str}' in measurements)) else VG*np.NaN
        IGlin=measurements[f'fIG@VD={VDlin_str}'] if (has_iglin:=(f'fIG@VD={VDlin_str}' in measurements)) else VG*np.NaN
        if IDsat.shape[1]==1 and np.isnan(IDsat[0]): has_idsat=False
        if IDlin.shape[1]==1 and np.isnan(IDlin[0]): has_idlin=False
        if IGsat.shape[1]==1 and np.isnan(IGsat[0]): has_igsat=False
        if IGlin.shape[1]==1 and np.isnan(IGlin[0]): has_iglin=False

        # Requirements on VG
        assert np.sum(np.abs(np.diff(VG,axis=0)))==0, "Might assume all rows of VG are same for uniform meas"
        VG1d=VG[0,:]
        if self.vgoff is not None:
            ind0=np.argmax(VG1d==self.vgoff)
            if not VG1d[ind0]==self.vgoff:
                logger.warning(f"Must be an exactly {self.vgoff} entry in VG, no tol for this")
                ind0=False
        else: ind0=False
        #indons={}
        #for k,v in self.Vgons.items():
        #    indons[k]=np.argmax(VG1d==v)
        #    if not np.allclose(VG1d[indons[k]],v):
        #        logger.warning(f"Must be an exactly {v} entry in VG for on-state")
        #        indons[k]=False
        DVG=VG1d[1]-VG1d[0]
        assert np.allclose(np.diff(VG),DVG), "VG should be even spacing"
        assert np.sign(DVG)==(-1 if self.pol=='p' else 1), "VG should sweep off-to-on"

        if has_idsat:
            gm=savgol_filter(IDsat,*gmsavgol,deriv=1)/DVG
            invswing=savgol_filter(np.log10(np.abs(IDsat.T/W).T+self.Iswf),*sssavgol,deriv=1)/np.abs(DVG)
        else:
            gm=VG*np.NaN
            invswing=VG*np.NaN
        if has_idlin:
            invswing_lin=savgol_filter(np.log10(np.abs(IDlin.T/W).T+self.Iswf),*sssavgol,deriv=1)/np.abs(DVG)
        else:
            invswing_lin=VG*np.NaN

        all_inds=np.arange(len(VG))
        inds_gmpeak=np.argmax(gm,axis=1)
        gmpeak=gm[all_inds,inds_gmpeak]
        v_gmpeak=VG1d[inds_gmpeak]
        i_gmpeak=abs(IDsat[all_inds,inds_gmpeak])
        vt_gmpeak=v_gmpeak-np.sign(DVG)*i_gmpeak/gmpeak

        measurements['Ion [A]']=np.abs(IDsat[:,-1])
        measurements['Ion_lin [A]']=np.abs(IDlin[:,-1])
        measurements['Ioff [A]']=measurements['Ion [A]']*np.NaN if (ind0 is False) or (not has_idsat) else np.abs(IDsat[:,ind0])
        measurements['Ioff_lin [A]']=measurements['Ion [A]']*np.NaN if (ind0 is False) or (not has_idlin) else np.abs(IDlin[:,ind0])
        measurements['Ioffmin [A]']=np.min(np.abs(IDsat),axis=1)
        measurements['Ioffstart [A]']=np.abs(IDsat[:,0])
        measurements['Ioffstart_lin [A]']=np.abs(IDlin[:,0])
        measurements['Ion/Ioff']=measurements['Ion [A]']/measurements['Ioff [A]']
        measurements['Ion/Ioffmin']=measurements['Ion [A]']/measurements['Ioffmin [A]']
        measurements['Ion/Ioffstart']=measurements['Ion [A]']/measurements['Ioffstart [A]']
        #for k,ind in indons.items():
        #    measurements[f'Ron{k} [ohm]']=measurements['Ion [A]']*np.NaN if (ind is False)# or (not has_idlin) else np.abs(VDlin)/(np.abs(IDlin[:,ind])+tol)
        #    measurements[f'RonW{k} [ohm.um]']=measurements[f'Ron{k} [ohm]']*W*1e6
        for k,v in self.Vgons.items():
            measurements[f'Ron{k} [ohm]']=np.abs(VDlin)/np.abs(YatX(X=VG,Y=IDlin,x=v))
            measurements[f'RonW{k} [ohm.um]']=measurements[f'Ron{k} [ohm]']*W*1e6
        measurements[f'Ronstop [ohm]']=np.abs(VDlin)/(measurements['Ion_lin [A]']+tol)
        measurements[f'RonWstop [ohm.um]']=measurements[f'Ronstop [ohm]']*W*1e6
        measurements['VGstop [V]']=VG1d[-1]
        for k,v in self.Iccs.items():
            measurements[f'VTcc{k}_lin']=VTCC((IDlin.T/W).T,VG,v,itol=tol)
            measurements[f'VTcc{k}_sat']=VTCC((IDsat.T/W).T,VG,v,itol=tol)
            measurements[f'DIBL{k} [mV/V]']=\
                -1000*(measurements[f'VTcc{k}_sat']-measurements[f'VTcc{k}_lin'])/(VDsat-VDlin)
        measurements['VTgm_sat']=vt_gmpeak
        measurements['GM_peak [S]']=gmpeak
        measurements['SS [mV/dec]']=1e3/np.max(invswing,axis=1)
        measurements['SS_lin [mV/dec]']=1e3/np.max(invswing_lin,axis=1)
        measurements['Igoffstart [A]']=np.abs(IGsat[:,0])
        measurements['Igoffstart_lin [A]']=np.abs(IGlin[:,0])
        measurements['Igonstop [A]']=np.abs(IGsat[:,-1])
        measurements['Igonstop_lin [A]']=np.abs(IGlin[:,-1])
        measurements['Igmax_lin [A]']=np.max(np.abs(IGlin),axis=1)
        measurements['Igmax_sat [A]']=np.max(np.abs(IGsat),axis=1)
        measurements['Igmax [A]']=np.maximum(measurements['Igmax_lin [A]'],measurements['Igmax_sat [A]'])


@dataclasses.dataclass
class IdVd(MeasurementType):
    norm_column: str
    pol: str = 'n'
    VGoffs: dict[str,float] = dataclasses.field(default_factory=lambda:{'':0})
    VDDs: dict[str,float] = dataclasses.field(default_factory=lambda:{'':1})

    def __post_init__(self):
        self._norm_col_units={'mm':1e-3,'um':1e-6,'nm':1e-9} \
            [self.norm_column.split("[")[1].split("]")[0]]

    def get_norm(self, measurements):
        return np.array(measurements[self.norm_column],dtype=np.float32)*self._norm_col_units

    def get_preferred_dtype(self,header):
        return np.float32

    def analyze(self, measurements):
        has_ig=any('IG' in k for k in measurements.headers)
        VGstrs=[k.split("=")[-1] for k in measurements.headers if k.startswith('fID')]
        for VDDlabel,VDD in self.VDDs.items():
            for VGofflabel,VGoff in self.VGoffs.items():
                try: VGoffstr=only([k for k in VGstrs if np.isclose(float(k),VGoff)])
                except: measurements[f'Ileak{VGofflabel}{VDDlabel} [A]']= np.NaN
                else: measurements[f'Ileak{VGofflabel}{VDDlabel} [A]']= \
                        YatX(X=measurements['VD'],Y=measurements[f'fID@VG={VGoffstr}'],x=VDD)
        if has_ig:
            Igmax=np.max(np.vstack([np.max(np.abs(measurements[f'fIG@VG={vgs}']),axis=1) for vgs in VGstrs]).T,axis=1)
        else: Igmax=np.NaN
        measurements['Igmax [A]']=Igmax

    def __str__(self):
        return 'IdVd'



@dataclasses.dataclass
class KelvinRon(MeasurementWithLinearNormColumn):
    # Columns should include 'VG', 'fRon@ID=...', 'fVSSense@ID=...', 'fVD2p@ID=...', 'fVDSense@ID=...'

    only_ats:Optional[list[str]] = None
    only_fields:Optional[list[str]] = None
    main_ron_id:Optional[str] = None
    #vg_for_ron: Optional[float] = 1.5
    VGons: dict[str,float] = dataclasses.field(default_factory=lambda:{})

    def __post_init__(self):
        super().__post_init__()
        if self.only_ats is not None: raise NotImplementedError("Only ats not implemented")
        if self.only_fields is not None: raise NotImplementedError("Only fields not implemented")

    def get_preferred_dtype(self,header):
        return np.float32

    def analyze(self, measurements,rexts=None):

        W=self.get_norm(measurements)

        # TODO: need less janky mechanism for adding to headers
        if rexts is not None:
            rsext=rexts.scalar_table_with_layout_params(['xtor']) \
                [['xtor','Rs_ext [ohm]']].groupby(['xtor']).median()
            df=measurements.scalar_table_with_layout_params(['xtor'])
            measurements['Rs_ext [ohm]']=pd.merge(left=df,right=rsext,
                                                  on=['xtor'],how='left',validate='m:1')['Rs_ext [ohm]']

            id_strs=[h.split("=")[1] for h in measurements.headers if 'fRon@ID=' in h]
            for id in id_strs:
                measurements._the_dataframe[f'VGSi@ID={id}']= \
                    list((measurements[f'VG'].T-float(id)*np.array(measurements['Rs_ext [ohm]'])).T)
                if f'VGSi@ID={id}' not in measurements.headers: measurements.headers.append(f'VGSi@ID={id}')
        else:
            for k in measurements.headers:
                VS=measurements[k]
                if 'SourceSense' in k:
                    if k.replace("VSourceSense","VGSi") not in measurements.headers:
                        measurements.headers.append(k.replace("VSourceSense","VGSi"))
                    measurements._the_dataframe[k.replace("VSourceSense","VGSi")]=measurements['VG']-VS

        fron_headers=[h for h in measurements.headers if 'fRon@ID=' in h]
        if self.main_ron_id is None:
            if len(fron_headers)==1:
                main_ron_id=fron_headers[0].split("=")[-1]
            else: main_ron_id=None
        else: main_ron_id=self.main_ron_id
        if (k:=('fRon@ID='+str(main_ron_id))) in measurements.headers:
            #if self.vg_for_ron is not None:
            #    ind_vg=np.argmin(np.abs(measurements['VG']-self.vg_for_ron))
            #    assert np.allclose(self.vg_for_ron,measurements['VG'][:,ind_vg],atol=1e-3)
            #else: ind_vg=-1
            measurements['Ronstop [ohm]']=measurements[k][:,-1]
            for vglabel, vgon in self.VGons.items():
                measurements[f'Ron{vglabel} [ohm]']=YatX(measurements['VG'],measurements[k],vgon)
                if W is not None:
                    measurements[f'Ron{vglabel}W [ohm.um]']=measurements[f'Ron{vglabel} [ohm]']*W*1e6
            if any('Sense' in k for k in measurements.headers):
                measurements['Rs_ext [ohm]']= \
                    measurements['fVSSense@ID='+str(main_ron_id)][:,-1] / float(main_ron_id)
                measurements['Rd_ext [ohm]']= \
                    (measurements['fVD2p@ID='+str(main_ron_id)][:,-1]
                     -measurements['fVDSense@ID='+str(main_ron_id)][:,-1]) / float(main_ron_id)
            #measurements['RonZ [Ωμm]']=measurements['Ron [Ω]']*measurements['Znorm [nm]']/1e3


    @staticmethod
    def VTRon(VG:np.ndarray,Ron:np.ndarray):
        # Compute the effective on-state VT from the slope of the conductance at peak transconductance
        dVG=VG[0,1]-VG[0,0] # Assumes uniform and regular VG
        G=np.clip(1/Ron,1e-12,1e6)
        dG_dVG=savgol_filter(G,3,1,deriv=1)/dVG
        try:
            ind_peak=np.nanargmax(dG_dVG,axis=-1)
        except:
            import pdb; pdb.set_trace()
            raise
        dG_dVG_peak=dG_dVG[np.arange(len(ind_peak)),ind_peak]
        G_peak=G[np.arange(len(ind_peak)),ind_peak]
        VG_peak=VG[0,ind_peak]
        VT=VG_peak-G_peak/dG_dVG_peak
        return VT

    def __str__(self):
        return 'KelvinRon'
