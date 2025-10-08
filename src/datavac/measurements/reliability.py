from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
from datavac.config.data_definition import DVColumn
from datavac.measurements.measurement_group import ExtractionAddon
from datavac.util.dvlogging import logger
from datavac.util.util import asnamedict

if TYPE_CHECKING:
    from datavac.io.measurement_table import MultiUniformMeasurementTable

@dataclass
class ConvertToT0DeltasAddon(ExtractionAddon):
    variable: str
    offset_or_percent: str = 'offset'
    device_grouping_cols: Sequence[str] = ('FileName', 'FQSite')
    def additional_extract_by_mumt(self, mumt:MultiUniformMeasurementTable, **kwargs):
        import numpy as np
        logger.debug("In convert_to_t0_deltas")
        df=mumt.scalar_table.copy()
        variable=self.variable

        df[f'{variable}0']=np.nan
        for k,grp in df.groupby(list(self.device_grouping_cols)):
            # If you can guarantee only one measurement at t=0
            #df.loc[grp.index,f'{variable}0']=only_row(grp[grp['total_stress_time']==0])[variable]
            # Or we could use the last measurement at t=0
            lastt0meas=grp[grp['total_stress_time']==0]['take'].idxmax()
            df.loc[grp.index,f'{variable}0']=grp.loc[lastt0meas,variable]

        mumt[f'{variable}0']=df[f'{variable}0']
        match self.offset_or_percent:
            case 'offset':
                mumt[f'Delta{variable}']=df[variable]-df[f'{variable}0']
            case '-offset':
                mumt[f'-Delta{variable}']=df[f'{variable}0']-df[variable]
            case 'percent':
                mumt[f'%Delta{variable.split(" [")[0]}']=100*(df[variable]-df[f'{variable}0'])/df[f'{variable}0']
            case _:
                raise Exception(f"What is {self.offset_or_percent}?")
            
    def additional_available_extr_columns(self) -> dict[str, DVColumn]:
        variable=self.variable
        added_colums= [DVColumn(f'{variable}0', 'float', f'{variable} at t=0'),]
        if self.offset_or_percent == 'offset':
            added_colums.append(DVColumn(f'Delta{variable}', 'float', f'{variable} diference from value at t=0'))
        elif self.offset_or_percent == '-offset':
            added_colums.append(DVColumn(f'-Delta{variable}', 'float', f'negative {variable} diference from value at t=0'))
        elif self.offset_or_percent == 'percent':
            added_colums.append(DVColumn(f'%Delta{variable.split(" [")[0]}', 'float', f'{variable} percent difference from value at t=0'))
        return asnamedict(*added_colums)