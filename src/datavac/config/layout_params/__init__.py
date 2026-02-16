from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, cast






if TYPE_CHECKING:
    from sqlalchemy import Connection
    import pandas as pd
    from datavac.measurements.measurement_group import MeasurementGroup

class LayoutParameters:

    drop_col_when_merging_with_layout_params: Callable[[str],bool] = (lambda self, c: c.startswith("PAD:"))

    _tables_by_meas: dict[str, pd.DataFrame] = {}

    def __init__(self, force_regenerate: bool = False):
        super().__init__()

    def merge_with_layout_params(self, meas_df:pd.DataFrame, for_measurement_group: MeasurementGroup,
                                 param_names: Optional[list[str]] = None, on_missing: str = 'error'):
        #structures_to_get=meas_df['Structure'].unique()
        #params=self.get_params(structures_to_get,allow_partial=False,for_measurement_group=for_measurement_group)
        from typing import cast
        import pandas as pd
        from datavac.measurements.measurement_group import SemiDevMeasurementGroup
        layout_param_group:Optional[str] = cast(SemiDevMeasurementGroup,for_measurement_group).layout_param_group
        assert layout_param_group is not None, \
            f"No layout parameter group specified for measurement group {for_measurement_group.name}"
        if layout_param_group not in self._tables_by_meas:
            raise Exception(f"No layout parameters for measurement group {for_measurement_group}")
        params=self._tables_by_meas[layout_param_group]
        if param_names:
            params=params[[pn for pn in param_names if pn in params.columns]].copy()
            for param in param_names:
                if param not in params.columns:
                    if on_missing=='error':
                        raise Exception(f"Missing parameter {param}")
                    elif on_missing=='NA':
                        params[param]=pd.NA
                    elif on_missing=='ignore':
                        continue
                    else:
                        raise Exception(f"Unrecognized value for on_missing={on_missing}")
        else:
            # Warning: this will silently allow columns in meas_df to override columns in params
            cols_to_drop=[c for c in params if c in meas_df]\
                +[c for c in params if self.drop_col_when_merging_with_layout_params(c)]
            params=params.drop(columns=cols_to_drop)
        
        # TODO: generalize the choice of merge column
        left_on='Structure' if 'Structure' in meas_df.columns else 'Site'
        merged=pd.merge(left=meas_df,right=params,how='left',left_on=[left_on],right_index=True,
                        suffixes=(None,'_param'))
        return merged

    #def get_params(self,structures,mask,drop_pads=True,for_measurement_group=None,allow_partial=False):
    def get_params(self,structures: list[str], mask: Optional[str] = None, drop_pads: bool = True,
                   for_measurement_group: Optional[MeasurementGroup] = None, allow_partial: bool = False) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented in a subclass")

def LP(force_regenerate:bool=False) -> LayoutParameters:
    from datavac.config.data_definition import SemiDeviceDataDefinition
    from datavac.config.data_definition import DDEF
    return cast(SemiDeviceDataDefinition,DDEF()).layout_params_func(force_regenerate=force_regenerate)