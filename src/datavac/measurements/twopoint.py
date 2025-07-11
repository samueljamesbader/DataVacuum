from typing import Optional, cast
from datavac.config.data_definition import DDEF, DVColumn, HigherAnalysis
from datavac.io.measurement_table import UniformMeasurementTable
from datavac.util.dvlogging import logger
from datavac.measurements.measurement_group import SemiDevMeasurementGroup
from datavac.util.util import asnamedict, only


import dataclasses

from pandas.core.api import DataFrame as DataFrame



@dataclasses.dataclass(eq=False, repr=False)
class IV(SemiDevMeasurementGroup):

    @staticmethod
    def get_preferred_dtype(header):
        import numpy as np
        return np.float32

    raw_vname: Optional[str] = None
    raw_iname: Optional[str] = None

    dirlist=[]
    V=None
    I=None

    def process_from_raw(self,raw_data_dict):

        if self.raw_vname is None:
            vcandidates=[k for k in raw_data_dict if k[0]=='V']
            raw_vname=only(vcandidates)

        if self.raw_iname is None:
            ifcandidates=[k for k in raw_data_dict if k[0]!='V' and k.split("@")[0][-1]==vcandidates[0][-1] \
                          and (k.startswith("I") or k.startswith("f_I") or k.startswith("fI"))]
            raw_iname=only(ifcandidates)

        return {'V':raw_data_dict[raw_vname],'I':raw_data_dict[raw_iname]}


class OpenIV(IV):
    def extract_by_umt(self, measurements:UniformMeasurementTable):
        import numpy as np
        I=measurements['I']
        V=measurements['V']
        measurements['Imax [A]']=np.max(np.abs(I),axis=1)


@dataclasses.dataclass(eq=False, repr=False)
class ResistorIV(IV):
    R_if_bad: Optional[float] = None
    Vmax: Optional[float] = None
    Imin: float = 1e-12

    def extract_by_umt(self, measurements:UniformMeasurementTable):
        import numpy as np
        import pandas as pd
        from scipy.stats import linregress
        I=measurements['I']
        V=measurements['V']

        if self.Vmax is not None:
            ptmask=np.logical_and(V[0]!=0,V[0]<self.Vmax)
        else:
            ptmask=V[0]!=0

        G=I[:,ptmask]/V[:,ptmask]
        G[abs(I[:,ptmask])<self.Imin]=np.NaN
        G[G<0]=np.NaN
        Gmean=np.mean(G,axis=1)
        Gmean[np.sum(np.logical_not(np.isclose(G.T,Gmean,rtol=.1).T),axis=1)!=0]=np.NaN
        R=1/Gmean

        measurements['R']=R
        if self.R_if_bad is not None: measurements.s['R']=measurements.s['R'].fillna(self.R_if_bad).astype('float64')
        try:
            if 'para_ser' in (mtab:=measurements.scalar_table_with_layout_params(['para_ser'],on_missing='ignore')):
                measurements['Runit']=measurements['R']*mtab['para_ser']
        except Exception as e:
            if "No layout parameters" in str(e): pass
            else: raise e
        measurements['Imax [A]']=np.max(np.abs(I[:,ptmask]),axis=1)

        measurements['Vmax_analysis']=self.Vmax or np.inf
    def available_extr_columns(self) -> dict[str, DVColumn]:
        return {**super().available_extr_columns(),
                **asnamedict(DVColumn('R', 'float64','Resistance in ohm')), # TODO: Change this to 'R [ohm]'
                **asnamedict(DVColumn('Imax [A]', 'float64','Maximum current in A')),
                **asnamedict(DVColumn('Vmax_analysis', 'float64','Maximum voltage used for analysis in V'))}

@dataclasses.dataclass(eq=False, repr=False)
class TLMIV(ResistorIV):
    Vmax: Optional[float] = .2

@dataclasses.dataclass(eq=False, repr=False, kw_only=True)
class TLMSummary(HigherAnalysis):
    outer_grouping: list[str] = dataclasses.field(default_factory=lambda: ['DieXY'])
    short_grouping: list[str] = dataclasses.field(default_factory=list)
    Z_column: str = 'Wtot [nm]'
    Lsep_column: str = 'Lsep [nm]'
    half_Lsep_column: Optional[str] = None
    parallel_column: Optional[str] = None
    series_column: Optional[str] = None
    Iopen: float = 1e-9
    R2min: float = .98
    R2min_rvs: float = .9
    allow_missing_short: bool = True
    subtract_short: bool = False
    keep_columns: list[str] = dataclasses.field(default_factory=list)
    filter_query: Optional[str] = None
    subsample_reference_names: dataclasses.InitVar[list[str]] = ['Dies']
    def __post_init__(self, *args, **kwargs):
        if self.half_Lsep_column is not None and self.Lsep_column is None:
            self.Lsep_column = f"2{self.half_Lsep_column}"
        if hasattr(super(),'__post_init__'): return super().__post_init__(*args, **kwargs) # type: ignore
    def analyze(self, tlm_umt:UniformMeasurementTable) -> DataFrame:
        """Analyzes TLM data and returns a summary DataFrame."""
        tlm_data= tlm_umt.scalar_table_with_layout_params(
            list(set([c for c in [self.Z_column, self.Lsep_column, self.half_Lsep_column,
                                  self.parallel_column, self.series_column,'devtype', # TODO: generalize or document devtype
                                  *self.outer_grouping, *self.short_grouping, *self.keep_columns]
               if c is not None])),on_missing='ignore')
        if self.half_Lsep_column is not None:
            assert self.Lsep_column not in tlm_data.columns, \
                f'Seems half_Lsep_column ("{self.half_Lsep_column}") is specified to generate Lsep_column "{self.Lsep_column}", '\
                f'but "{self.Lsep_column}" is already in the data.  If half_Lsep_column is not None, Lsep_column should not exist yet.'
            tlm_data=tlm_data.assign(**{self.Lsep_column: tlm_data[self.half_Lsep_column] * 2})
        if self.filter_query is not None:
            tlm_data=tlm_data.query(self.filter_query,engine='python')
        return tlm_summary(tlm_data,
                           outer_grouping=self.outer_grouping,
                           short_grouping=self.short_grouping,
                           Z_column=self.Z_column, Lsep_column=self.Lsep_column,
                           parallel_column=self.parallel_column, series_column=self.series_column,
                           Iopen=self.Iopen, R2min=self.R2min, R2min_rvs=self.R2min_rvs,
                           allow_missing_short=self.allow_missing_short, subtract_short=self.subtract_short,
                           keep_columns=self.keep_columns)
    def available_analysis_columns(self) -> dict[str, DVColumn]:
        mg = cast(SemiDevMeasurementGroup,DDEF().measurement_groups[only(self.required_dependencies.keys())])
        avec=mg.available_extr_columns()
        cols_avail_to_keep = dict( # type: ignore
            **asnamedict(*mg.meas_columns),
            **{c:avec[c] for c in mg.extr_column_names},
            **{c.name:c for ssr_name in mg.subsample_reference_names for c in DDEF().subsample_references[ssr_name].info_columns},
            **asnamedict(*mg.trove().load_info_columns)
            )
        cols_avail_to_keep = {k:v for k,v in cols_avail_to_keep.items() if k!=self.Lsep_column}
        columns_kept = {cn:c for cn,c in cols_avail_to_keep.items() if cn in
                        [*self.outer_grouping,*self.short_grouping,self.Z_column,self.Lsep_column,
                         self.parallel_column,self.series_column,*self.keep_columns] }
        return {**super().available_analysis_columns(),
                **asnamedict(DVColumn('Rc [ohm]', 'float64', 'Contact resistance in ohm')),
                **asnamedict(DVColumn('RcZ [ohm.um]', 'float64', 'Contact resistance per unit width in ohm.um')),
                **asnamedict(DVColumn('Rdrift [ohm/um]', 'float64', 'Drift resistance per unit width in ohm/um')),
                **asnamedict(DVColumn('Rsh [ohm]', 'float64', 'Sheet resistance in ohm')),
                **asnamedict(DVColumn('LT', 'float64', 'Lateral TLM length in um')),
                **asnamedict(DVColumn('ρc [ohm.cm^2]', 'float64', 'Contact resistivity in ohm.cm^2')),
                **asnamedict(DVColumn('goodfit', 'boolean', 'Whether the fit was good')),
                **columns_kept} 

def tlm_summary(tab,
                outer_grouping,
                short_grouping,
                Z_column,Lsep_column,parallel_column=None, series_column=None,
                do_plot=False,Iopen=1e-9,R2min=.98,R2min_rvs=.9,
                allow_missing_short=False,subtract_short=False,
                keep_columns=[]):
    import numpy as np
    import pandas as pd
    from scipy.stats import linregress
    assert len(set(outer_grouping+short_grouping+keep_columns))==len(outer_grouping+short_grouping+keep_columns),\
        f"Overlap between outer {outer_grouping} short {short_grouping} and keep {keep_columns}"
    # TODO: should be able to speed this up DRAMATICALLY with the multiy_singlex_linregress function
    # TODO: clean and add unit tests
    def unit_getter(header):
        unit_section=header.split()[-1]
        assert unit_section.startswith("[") and unit_section.endswith("]")
        unit=unit_section[1:-1]
        return {'um':1e-6,'nm':1e-9}[unit]
    output=[]
    for outer_key,dgrp in tab.groupby(outer_grouping):
        if type(outer_key)==str:
            outer_key=(outer_key,)

        rshort:float
        if (not len(short_grouping)) or 'devtype' not in dgrp.keys():
            assert allow_missing_short
            gb=[[(),dgrp]]
            rshort=0.0
        else:
            gb=dgrp.query("devtype=='TLM'").groupby(short_grouping)
            r=dgrp.query("`devtype`=='short'")
            if len(r)>1:
                logger.warning("Averaging multiple short structures")
            if len(r)==0 and allow_missing_short:
                rshort=0.0
            else:
                #shortdata=only_row(r,f"No short structures associated with {outer_grouping} = {outer_key}")
                assert len(r), f"No short structures associated with {outer_grouping} = {outer_key}"

                shortdata=r
                rshort=shortdata.R.mean()

                # if the short is bad, skip the die
                if (rshort>100) or not (rshort==rshort):
                    logger.warning(f"Bad short for {outer_grouping} = {outer_key}")
                    continue
        for inner_key,grp in gb:
            Z=grp.iloc[0][Z_column]*unit_getter(Z_column)

            if do_plot:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(7,3))
                plt.subplot(121)

            cpara=grp[parallel_column] if parallel_column else 1
            cseri=grp[series_column] if series_column else 1
            Rs=list(grp.R*cpara/cseri)
            if do_plot:
                for i,raw in grp.iterrows():
                    if do_plot:
                        plt.plot(raw.V,raw.I/1e-6,label=raw[Lsep_column],linestyle=('dashed' if np.isnan(raw.R) else '-'))

            if do_plot:
                plt.suptitle(f"{outer_key}: {inner_key}")
                ylim=plt.ylim()
                if rshort!=0.0:
                    for _,sdr in shortdata.iterrows():
                        plt.plot(sdr.V,sdr.I/1e-6,'--')
                plt.ylim(ylim)
                plt.xlabel("V [V]")
                plt.ylabel("I [uA]")
                plt.axvline(0,color='k',linewidth=.5)
                plt.axhline(0,color='k',linewidth=.5)
                plt.legend(loc='best')


            Lseps=grp[Lsep_column]*unit_getter(Lsep_column)
            if len(set(Lseps))==1: continue
            slope:float;intercept:float;rval:float;
            slope,intercept,rval,*_=linregress(list(Lseps),Rs) # type: ignore
            goodfit=(rval**2>R2min_rvs)
            if not rval**2>R2min_rvs:
                #print(f"R^2={rval**2} too low for {str(outer_key+inner_key)}")
                slope,intercept=np.NaN,np.NaN
            if subtract_short:
                intercept-=rshort
            Rc=intercept/2
            Rsh=slope*Z
            LT=Rc*Z/Rsh
            ρc=Rc*Z*LT
            output.append(dict(**{'Rc [ohm]':Rc,'RcZ [ohm.um]':Rc*Z*1e6,
                                  'Rdrift [ohm/um]':Rsh/(Z*1e6),'Rsh [ohm]':Rsh,
                                  'LT':LT,'ρc [ohm.cm^2]':ρc*1e4,'goodfit':goodfit},
                               **dict(zip(outer_grouping,outer_key)),
                               **dict(zip(short_grouping,inner_key)),
                               **dict(zip(keep_columns,[grp[k].iloc[0] for k in keep_columns])),
                               **{'Wtlm [um]':grp[Z_column].iloc[0]*unit_getter(Z_column)*1e6,'Rshort':rshort},
                               )
                          )


            if do_plot:
                plt.subplot(122)
                plt.plot(Lseps/1e-9,Rs,'.')
                if not subtract_short:
                    plt.axhline(y=rshort,linestyle='dashed')
                plt.plot(Lseps/1e-9,Lseps*slope+intercept)
                plt.xlabel("Lsep [nm]")
                plt.ylabel("R [ohm]")

                plt.ylim(0)
                plt.xlim(0)
                plt.tight_layout()
        #print("")
    #import pdb; pdb.set_trace()
    try:
        output=pd.DataFrame(output).astype(dtype={k:tab[k].dtype for k in keep_columns+outer_grouping+short_grouping})
    except:
        raise
    #print(f"Number of successfully TLM'd dies: {len(output)}")
    #.set_index(outer_grouping+short_grouping)
    return output