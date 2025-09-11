from __future__ import annotations
import dataclasses
from typing import Optional, Union, cast, TYPE_CHECKING

from datavac.config.data_definition import DVColumn
from datavac.measurements.measurement_group import SemiDevMeasurementGroup

from datavac.util.util import asnamedict, only
from datavac.util.dvlogging import logger

if TYPE_CHECKING:
    from datavac.io.measurement_table import MeasurementTable, UniformMeasurementTable
    import numpy as np

@dataclasses.dataclass(eq=False,repr=False)
class MeasurementWithLinearNormColumn(SemiDevMeasurementGroup):
    norm_column: str = None
    def __post_init__(self):
        if self.norm_column is not None:
            self._norm_col_units={'mm':1e-3,'um':1e-6,'nm':1e-9} \
                [self.norm_column.split("[")[1].split("]")[0]]

    def get_norm(self, measurements: MeasurementTable):
        import numpy as np
        assert self.norm_column is not None
        #if self.norm_column is None:
        #    return None
        return np.array(measurements \
                        .scalar_table_with_layout_params(params=[self.norm_column],on_missing='ignore')[self.norm_column],dtype=np.float32) \
            *self._norm_col_units

@dataclasses.dataclass(eq=False,repr=False)
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
    Vgons: dict[str,float] = None #dataclasses.field(default_factory=lambda:{'':1})
    abs_vdlin: float = None
    abs_vdsat: float = None

    def __post_init__(self):
        super().__post_init__()
        if self.Vgons is None:
            self.Vgons={'':(1 if self.pol=='n' else -1)}

    def available_extr_columns(self) -> dict[str, DVColumn]:
        # Use abs_vdlin/abs_vdsat if provided, else describe as |VD|=max or |VD|=min, and use sign if pol is set
        if self.abs_vdsat is not None:
            vdsat_val = self.abs_vdsat if self.pol == 'n' else -self.abs_vdsat
            VDsat_desc = f"VD={vdsat_val} (saturation)"
        else:
            VDsat_desc = "|VD|=max (saturation)"
        if self.abs_vdlin is not None:
            vdlin_val = self.abs_vdlin if self.pol == 'n' else -self.abs_vdlin
            VDlin_desc = f"VD={vdlin_val} (linear)"
        else:
            VDlin_desc = "|VD|=min (linear)"
        extr_columns = [*super().available_extr_columns().values(),
            DVColumn('SS [mV/dec]', 'float64', 'Subthreshold swing'),
            DVColumn('Ion [A]', 'float64', f'On-state drain current (saturation, {VDsat_desc})'),
            DVColumn('Ion/W [A/m]', 'float64', f'On-state drain current per width (saturation, {VDsat_desc})'),
            DVColumn('Ion_lin [A]', 'float64', f'On-state drain current (linear, {VDlin_desc})'),
            DVColumn('Ion_lin/W [A/m]', 'float64', f'On-state drain current per width (linear, {VDlin_desc})'),
            DVColumn('Ioff [A]', 'float64', f'Off-state drain current (saturation, {VDsat_desc})'),
            DVColumn('Ioff/W [A/m]', 'float64', f'Off-state drain current per width (saturation, {VDsat_desc})'),
            DVColumn('Ioff_lin [A]', 'float64', f'Off-state drain current (linear, {VDlin_desc})'),
            DVColumn('Ioff_lin/W [A/m]', 'float64', f'Off-state drain current per width (linear, {VDlin_desc})'),
            DVColumn('Ioffmin [A]', 'float64', 'Minimum off-state current'),
            DVColumn('Ioffstart [A]', 'float64', f'Start off-state current (saturation, {VDsat_desc})'),
            DVColumn('Ioffstart_lin [A]', 'float64', f'Start off-state current (linear, {VDlin_desc})'),
            DVColumn('Ion/Ioff', 'float64', 'On/Off current ratio'),
            DVColumn('Ion/Ioffmin', 'float64', 'On/Off-min current ratio'),
            DVColumn('Ion/Ioffstart', 'float64', 'On/Off-start current ratio'),
            DVColumn('Ronstop [ohm]', 'float64', f'Ron at VGstop (linear, {VDlin_desc})'),
            DVColumn('RonWstop [ohm.um]', 'float64', f'Ron*W at VGstop (linear, {VDlin_desc})'),
            DVColumn('VGstop [V]', 'float64', 'VG at stop'),
            DVColumn('VGstart [V]', 'float64', 'VG at start'),
            DVColumn('VTgm_sat', 'float64', f'VT at gm peak (saturation, {VDsat_desc})'),
            DVColumn('GM_peak [S]', 'float64', 'Peak transconductance'),
            DVColumn('SS_lin [mV/dec]', 'float64', f'Subthreshold swing (linear, {VDlin_desc})'),
            DVColumn('SSstart_lin [mV/dec]', 'float64', f'SS at start (linear, {VDlin_desc})'),
            DVColumn('Igoffstart [A]', 'float64', f'Gate off current (saturation, {VDsat_desc})'),
            DVColumn('Igoffstart_lin [A]', 'float64', f'Gate off current (linear, {VDlin_desc})'),
            DVColumn('Igonstop [A]', 'float64', f'Gate on current (saturation, {VDsat_desc})'),
            DVColumn('Igonstop_lin [A]', 'float64', f'Gate on current (linear, {VDlin_desc})'),
            DVColumn('Igmax_lin [A]', 'float64', f'Max gate current (linear, {VDlin_desc})'),
            DVColumn('Igmax_sat [A]', 'float64', f'Max gate current (saturation, {VDsat_desc})'),
            DVColumn('Igfwdmax_lin [A]', 'float64', f'Max forward gate current (linear, {VDlin_desc})'),
            DVColumn('Igmax [A]', 'float64', 'Max gate current'),
            DVColumn('Igmax/W [A/m]', 'float64', 'Max gate current per width'),
        ]
        # Add columns for each Vgons
        for k, v in self.Vgons.items():
            extr_columns.append(DVColumn(f'Ron{k} [ohm]', 'float64', f'Ron at VG={v} V (linear, {VDlin_desc})'))
            extr_columns.append(DVColumn(f'RonW{k} [ohm.um]', 'float64', f'Normalized Ron*W at VG={v} V (linear, {VDlin_desc})'))
        # Add columns for each Iccs
        for k, v in self.Iccs.items():
            extr_columns.append(DVColumn(f'VTcc{k}_lin', 'float64', f'VTcc (linear, {VDlin_desc}) at Icc={v}'))
            extr_columns.append(DVColumn(f'VTcc{k}_sat', 'float64', f'VTcc (saturation, {VDsat_desc}) at Icc={v}'))
            extr_columns.append(DVColumn(f'DIBL{k} [mV/V]', 'float64', f'DIBL at Icc={v} ({VDsat_desc} - {VDlin_desc})'))
        return asnamedict(*extr_columns)


    def extract_by_umt(self, measurements:UniformMeasurementTable) -> None:
        import numpy as np
        from scipy.signal import savgol_filter
        from datavac.util.maths import VTCC, YatX

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
        assert VDsat*VDlin>0, f"Oops, VDsat {VDsat} and VDlin {VDlin} have different signs"

        W=self.get_norm(measurements)
        VG: np.ndarray=measurements['VG'] # type: ignore
        IDsat:np.ndarray[float]=measurements[f'fID@VD={VDsat_str}'] if (has_idsat:=(f'fID@VD={VDsat_str}' in measurements)) else VG*np.nan # type: ignore
        IDlin:np.ndarray[float]=measurements[f'fID@VD={VDlin_str}'] if (has_idlin:=(f'fID@VD={VDlin_str}' in measurements)) else VG*np.nan # type: ignore
        IGsat:np.ndarray[float]=measurements[f'fIG@VD={VDsat_str}'] if (has_igsat:=(f'fIG@VD={VDsat_str}' in measurements)) else VG*np.nan # type: ignore
        IGlin:np.ndarray[float]=measurements[f'fIG@VD={VDlin_str}'] if (has_iglin:=(f'fIG@VD={VDlin_str}' in measurements)) else VG*np.nan # type: ignore
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
                #logger.warning(f"Must be an exactly {self.vgoff} entry in VG, no tol for this")
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
            gm=VG*np.nan
            invswing=VG*np.nan
        if has_idlin:
            invswing_lin=savgol_filter(np.log10(np.abs(IDlin.T/W).T+self.Iswf),*sssavgol,deriv=1)/np.abs(DVG)
        else:
            invswing_lin=VG*np.nan

        all_inds=np.arange(len(VG))
        inds_gmpeak=np.argmax(gm,axis=1)
        gmpeak=gm[all_inds,inds_gmpeak]
        v_gmpeak=VG1d[inds_gmpeak]
        i_gmpeak=abs(IDsat[all_inds,inds_gmpeak])
        vt_gmpeak=v_gmpeak-np.sign(DVG)*i_gmpeak/gmpeak

        measurements['Ion [A]']=np.abs(IDsat[:,-1])
        measurements['Ion/W [A/m]']=measurements['Ion [A]']/W
        measurements['Ion_lin [A]']=np.abs(IDlin[:,-1])
        measurements['Ion_lin/W [A/m]']=measurements['Ion_lin [A]']/W
        measurements['Ioff [A]']=measurements['Ion [A]']*np.nan if (ind0 is False) or (not has_idsat) else np.abs(IDsat[:,ind0])
        measurements['Ioff/W [A/m]']=measurements['Ioff [A]']/W
        measurements['Ioff_lin [A]']=measurements['Ion [A]']*np.nan if (ind0 is False) or (not has_idlin) else np.abs(IDlin[:,ind0])
        measurements['Ioff_lin/W [A/m]']=measurements['Ioff_lin [A]']/W
        measurements['Ioffmin [A]']=np.min(np.abs(IDsat),axis=1)
        measurements['Ioffstart [A]']=np.abs(IDsat[:,0])
        measurements['Ioffstart_lin [A]']=np.abs(IDlin[:,0])
        measurements['Ion/Ioff']=measurements['Ion [A]']/(np.abs(measurements['Ioff [A]'])+tol)
        measurements['Ion/Ioffmin']=measurements['Ion [A]']/(np.abs(measurements['Ioffmin [A]'])+tol)
        measurements['Ion/Ioffstart']=measurements['Ion [A]']/(np.abs(measurements['Ioffstart [A]'])+tol)
        #for k,ind in indons.items():
        #    measurements[f'Ron{k} [ohm]']=measurements['Ion [A]']*np.nan if (ind is False)# or (not has_idlin) else np.abs(VDlin)/(np.abs(IDlin[:,ind])+tol)
        #    measurements[f'RonW{k} [ohm.um]']=measurements[f'Ron{k} [ohm]']*W*1e6
        for k,v in self.Vgons.items():
            measurements[f'Ron{k} [ohm]']=np.abs(VDlin)/np.abs(YatX(X=VG,Y=IDlin,x=v,reverse_crossing=(self.pol=='p'))+tol)\
                if has_idlin else measurements['Ion [A]']*np.nan
            measurements[f'RonW{k} [ohm.um]']=measurements[f'Ron{k} [ohm]']*W*1e6
        measurements[f'Ronstop [ohm]']=np.abs(VDlin)/(measurements['Ion_lin [A]']+tol)
        measurements[f'RonWstop [ohm.um]']=measurements[f'Ronstop [ohm]']*W*1e6
        measurements['VGstop [V]']=VG1d[-1]
        measurements['VGstart [V]']=VG1d[0]
        for k,v in self.Iccs.items():
            measurements[f'VTcc{k}_lin']=VTCC((IDlin.T/W).T,VG,v,itol=tol)
            measurements[f'VTcc{k}_sat']=VTCC((IDsat.T/W).T,VG,v,itol=tol)
            measurements[f'DIBL{k} [mV/V]']=\
                -1000*(measurements[f'VTcc{k}_sat']-measurements[f'VTcc{k}_lin'])/(VDsat-VDlin)
        measurements['VTgm_sat']=vt_gmpeak
        measurements['GM_peak [S]']=gmpeak
        measurements['SS [mV/dec]']=1e3/np.max(invswing,axis=1)
        measurements['SS_lin [mV/dec]']=1e3/np.max(invswing_lin,axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sstartlin=1e3/invswing_lin[:,0]; sstartlin[sstartlin<0]=np.nan
        measurements['SSstart_lin [mV/dec]']=sstartlin
        measurements['Igoffstart [A]']=np.abs(IGsat[:,0])
        measurements['Igoffstart_lin [A]']=np.abs(IGlin[:,0])
        measurements['Igonstop [A]']=np.abs(IGsat[:,-1])
        measurements['Igonstop_lin [A]']=np.abs(IGlin[:,-1])
        measurements['Igmax_lin [A]']=np.max(np.abs(IGlin),axis=1)
        measurements['Igmax_sat [A]']=np.max(np.abs(IGsat),axis=1)

        fwd=(VG1d>0) if (self.pol=='n') else (VG1d<0)
        measurements['Igfwdmax_lin [A]']=np.max(np.abs(IGlin[:,fwd]),axis=1)

        if has_iglin == has_igsat:
            measurements['Igmax [A]']=np.maximum(measurements['Igmax_lin [A]'],measurements['Igmax_sat [A]'])
        elif has_iglin: measurements['Igmax [A]']=measurements['Igmax_lin [A]']
        elif has_igsat: measurements['Igmax [A]']=measurements['Igmax_sat [A]']
        measurements['Igmax/W [A/m]']=measurements['Igmax [A]']/W


@dataclasses.dataclass(eq=False,repr=False,kw_only=True)
class IdVd(SemiDevMeasurementGroup):
    norm_column: str
    pol: str = 'n'
    VGoffs: dict[str,float] = dataclasses.field(default_factory=lambda:{'':0})
    VDDs: dict[str,float] = dataclasses.field(default_factory=lambda:{'':1})

    def __post_init__(self):
        self._norm_col_units={'mm':1e-3,'um':1e-6,'nm':1e-9} \
            [self.norm_column.split("[")[1].split("]")[0]]

    def get_norm(self, measurements):
        import numpy as np
        return np.array(measurements[self.norm_column],dtype=np.float32)*self._norm_col_units

    def get_preferred_dtype(self,header):
        import numpy as np
        return np.float32

    def extract_by_umt(self, measurements:UniformMeasurementTable):
        import numpy as np
        from datavac.util.maths import YatX
        has_ig=any('IG' in k for k in measurements.headers)
        VGstrs=[k.split("=")[-1] for k in measurements.headers if k.startswith('fID')]
        for VGofflabel,VGoff in self.VGoffs.items():
            try: VGoffstr=only([k for k in VGstrs if np.isclose(float(k),VGoff)])
            except:
                for VDDlabel,VDD in self.VDDs.items():
                    measurements[f'Ileak{VGofflabel}{VDDlabel} [A]']= np.nan
                measurements[f'Idmax{VGofflabel} [A]']=np.nan
            else:
                for VDDlabel,VDD in self.VDDs.items():
                    measurements[f'Ileak{VGofflabel}{VDDlabel} [A]']= \
                        YatX(X=cast(np.ndarray,measurements['VD']),Y=cast(np.ndarray,measurements[f'fID@VG={VGoffstr}']),x=VDD)
                measurements[f'Idmax{VGofflabel} [A]']=np.max(np.abs(measurements[f'fID@VG={VGoffstr}']),axis=1)
        if has_ig:
            Igmax=np.max(np.vstack([np.max(np.abs(measurements[f'fIG@VG={vgs}']),axis=1) for vgs in VGstrs]).T,axis=1)
        else: Igmax=np.nan
        measurements['Igmax [A]']=Igmax
    def available_extr_columns(self) -> dict[str, DVColumn]:
        return {**super().available_extr_columns(),
                **asnamedict(
                    *[c for VGofflabel,VGoff in self.VGoffs.items()
                        for VDDlabel,VDD in self.VDDs.items()
                        for c in [
                            DVColumn(f'Ileak{VGofflabel}{VDDlabel} [A]', 'float64',
                                     f'Leakage current at VG={VGoff} V, VDD={VDD} V'),
                            DVColumn(f'Idmax{VGofflabel} [A]', 'float64',
                                     f'Max drain current at VG={VGoff} V'),
                      ]],
                      DVColumn('Igmax [A]', 'float64',
                               'Max gate current across all VG offsets and VDDs')
                )}

    def __str__(self):
        return 'IdVd'



@dataclasses.dataclass(eq=False,repr=False,kw_only=True)
class KelvinRon(MeasurementWithLinearNormColumn):
    # Columns should include 'VG', 'fRon@ID=...', 'fVSSense@ID=...', 'fVD2p@ID=...', 'fVDSense@ID=...'

    only_ats:Optional[list[str]] = None
    only_fields:Optional[list[str]] = None
    main_ron_id:Optional[str] = None
    #vg_for_ron: Optional[float] = 1.5
    VGons: dict[str,float] = dataclasses.field(default_factory=lambda:{})

    merge_rexts_on: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.only_ats is not None: raise NotImplementedError("Only ats not implemented")
        if self.only_fields is not None: raise NotImplementedError("Only fields not implemented")

    def get_preferred_dtype(self,header):
        import numpy as np
        return np.float32

    def extract_by_umt(self, measurements,rexts=None):
        import pandas as pd 
        import numpy as np
        from datavac.util.maths import YatX

        W=self.get_norm(measurements)

        # TODO: need less janky mechanism for adding to headers
        if (rexts is not None):
            assert self.merge_rexts_on is not None, "What to merge RExt on"
            rsext=rexts.scalar_table_with_layout_params(self.merge_rexts_on,on_missing='ignore') \
                [[*self.merge_rexts_on,'Rs_ext [ohm]']].groupby(self.merge_rexts_on).median()
            df=measurements.scalar_table_with_layout_params(self.merge_rexts_on,on_missing='ignore')
            measurements['Rs_ext [ohm]']=pd.merge(left=df,right=rsext,
                                                  on=self.merge_rexts_on,how='left',validate='m:1')['Rs_ext [ohm]']

            id_strs=[h.split("=")[1] for h in measurements.headers if 'fRon@ID=' in h]
            for id in id_strs:
                measurements._the_dataframe[f'VGSi@ID={id}']= \
                    list((measurements[f'VG'].T-float(id)*np.array(measurements['Rs_ext [ohm]'])).T)
                if f'VGSi@ID={id}' not in measurements.headers: measurements.headers.append(f'VGSi@ID={id}')
        else:
            assert self.merge_rexts_on is None, "This category requires RExt measurements"
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
        assert main_ron_id is not None
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
    def available_extr_columns(self) -> dict[str, DVColumn]:
        return {**super().available_extr_columns(),
                **asnamedict(
                    DVColumn('Ronstop [ohm]', 'float64', 'Ron at VGstop'),
                    DVColumn('VGstop [V]', 'float64', 'VG at stop'),
                    DVColumn('Rs_ext [ohm]', 'float64', 'External source resistance'),
                    DVColumn('Rd_ext [ohm]', 'float64', 'External drain resistance'),
                    *[c for vglabel, vgon in self.VGons.items()
                        for c in [
                            DVColumn(f'Ron{vglabel} [ohm]', 'float64', f'Ron at VG={vgon} V'),
                            DVColumn(f'Ron{vglabel}W [ohm.um]', 'float64', f'Normalized Ron*W at VG={vgon} V'),
                        ]])
        }


    @staticmethod
    def VTRon(VG:np.ndarray,Ron:np.ndarray):
        import numpy as np
        from scipy.signal import savgol_filter
        # Compute the effective on-state VT from the slope of the conductance at peak transconductance
        dVG=VG[0,1]-VG[0,0] # Assumes uniform and regular VG
        Ron=Ron.copy()
        Ron[Ron==0]=1e12
        G=np.clip(1/Ron,1e-12,1e6)
        G[np.isnan(G)]=1e-12
        dG_dVG=savgol_filter(G,3,1,deriv=1)/dVG
        ind_peak=np.nanargmax(dG_dVG,axis=-1)
        dG_dVG_peak=dG_dVG[np.arange(len(ind_peak)),ind_peak]
        G_peak=G[np.arange(len(ind_peak)),ind_peak]
        VG_peak=VG[0,ind_peak]
        VT=VG_peak-G_peak/dG_dVG_peak
        return VG_peak,VT

    def __str__(self):
        return 'KelvinRon'
