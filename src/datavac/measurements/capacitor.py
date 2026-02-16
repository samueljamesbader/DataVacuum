from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from datavac.config.data_definition import DVColumn
from datavac.measurements.measurement_group import SemiDevMeasurementGroup
from datavac.util.dvlogging import logger
from datavac.util.util import asnamedict

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from datavac.io.measurement_table import MeasurementTable, UniformMeasurementTable

@dataclass(eq=False,repr=False,kw_only=True)
class CapCV(SemiDevMeasurementGroup):

    def get_preferred_dtype(self,header):
        import numpy as np
        return np.float32

    def __str__(self):
        return 'C-V'
    
    open_match_cols:Sequence[str]=()
    open_filter_cols:Mapping[str,Any]=MappingProxyType({})
    open_aggregation_func:str='mean'
    
    @staticmethod
    def open_subtraction(cv:UniformMeasurementTable, opens: MeasurementTable|None,
                         open_match_cols: Sequence[str], open_filter_cols: Mapping[str,Any] = {}, open_aggregation_func:str='mean'):
        if opens is not None:
            import pandas as pd
            shcols=list(open_match_cols)+list(open_filter_cols.keys())
            opens_df=opens.scalar_table_with_layout_params(shcols,on_missing='ignore')
            for col, val in open_filter_cols.items():
                opens_df=opens_df[opens_df[col]==val]
            opens_df=opens_df[list(open_match_cols)+['Cmean [F]']].groupby(list(open_match_cols)).agg(open_aggregation_func).reset_index()
            ordered_opens=pd.merge(left=cv.scalar_table_with_layout_params(list(open_match_cols),on_missing='ignore')[list(open_match_cols)],
                                   right=opens_df, how='left',on=open_match_cols,validate='m:1')
            cv.s['Copen [F]']=ordered_opens['Cmean [F]']
        else: cv.s['Copen [F]']=0

    def extract_by_umt(self, measurements:UniformMeasurementTable, opens=None):
        import numpy as np
        import pandas as pd
        cv:MeasurementTable=measurements
        fstrs=[(k.split("@")[-1].split("=")[-1]) for k in cv.headers if 'fCp@freq=' in k]
        assert all([fstr.endswith('k') for fstr in fstrs])
        maxfstr=max(fstrs,key=lambda fstr:float(fstr[:-1]))
        maxf=float(maxfstr[:-1])*1e3
        self.open_subtraction(cv, opens, self.open_match_cols, self.open_filter_cols, self.open_aggregation_func)
        cv['Cmean [F]']=np.mean(cv[f'fCp@freq={maxfstr}'],axis=1)-cv['Copen [F]']

    def available_extr_columns(self) -> dict[str, DVColumn]:
        return {**super().available_extr_columns(),
                **asnamedict(
                    DVColumn('Cmean [F]','float','mean capacitance after open subtraction'),
                    DVColumn('Copen [F]','float','open capacitance used for subtraction'),
                )}