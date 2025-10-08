from dataclasses import InitVar, dataclass, field
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from datavac.config.data_definition import DDEF, DVColumn
from datavac.config.sample_splits import SampleSplitManager
import pandas as pd
from datavac.util.dvlogging import logger
if TYPE_CHECKING:
    import pandas as pd
    from openpyxl import Workbook


@dataclass
class FolderSampleSplitManager(SampleSplitManager):

    split_dir: Path = field(default_factory=lambda: Path(os.environ['DATAVACUUM_SPLIT_DIR']))
    _cached_column_info: dict[str, list[DVColumn]] = field(default_factory=dict, init=False)

    def get_flow_names_from_external(self) -> list[str]:
        # Read flow names from the split directory
        return [f.stem for f in self.split_dir.glob('*.xlsx') if f.is_file() and "~" not in f.stem]
    
    def _get_split_table_columns_and_wb(self, flow_name: str) -> tuple[list[DVColumn], 'Workbook']:
        from openpyxl import load_workbook
        import pandas as pd
        from datavac.config.data_definition import DDEF
        logger.debug(f"Loading split table {flow_name} from {self.split_dir/(flow_name+'.xlsx')}")
        wb=load_workbook(self.split_dir/(flow_name+'.xlsx'), data_only=True, read_only=True)
        ws = wb.active; assert ws is not None
        riter=iter(ws.rows)
        # Ignore "superheaders" row
        next(riter)
        # Get a dtypes dict
        headers=[str(c.value) for c in next(riter)]
        to_nullable={'string':'string', 'int':'Int64', 'integer': 'Int64', 'datetime':'datetime64[ns]', 'float': 'Float64'}
        dtypes= dict(zip(headers,[to_nullable.get(str(c.value),'IGNORE') for c in next(riter)]))
        descs= dict(zip(headers,[str(c.value) for c in next(riter)]))
        
        self._cached_column_info[flow_name]=\
            [DVColumn(name=col, pd_dtype=dtypes[col], description=descs[col])
             for col in headers if col not in DDEF().ALL_SAMPLE_COLNAMES and ('IGNORE' not in dtypes[col].upper())]
        return self._cached_column_info[flow_name], wb
    
    def get_split_table_columns_from_external(self, flow_name: str) -> list[DVColumn]:
        if flow_name in self._cached_column_info:
            return self._cached_column_info[flow_name]
        else:
            columns, _ = self._get_split_table_columns_and_wb(flow_name)
            return columns
    
    def get_split_table_from_external(self, flow_name: str) -> pd.DataFrame:
        columns, wb = self._get_split_table_columns_and_wb(flow_name)
        ws= wb.active; assert ws is not None
        dtypes = {col.name: col.pd_dtype for col in columns}
        df=pd.read_excel(wb, sheet_name=ws.title, skiprows=[0, 2, 3], engine='openpyxl',
                         dtype=dtypes, na_values=['','???'],keep_default_na=False)
        return df[[DDEF().SAMPLE_COLNAME,*[c for c in df.columns if c!=DDEF().SAMPLE_COLNAME]]]


def check_against_extra_source_and_combine(df_manual: pd.DataFrame, df_extra: pd.DataFrame,
                                           all_columns: Optional[list[str]]=None,
                                           flow_name: str='manual') -> pd.DataFrame:
    """
    Check the manual split table against the extra split data and combine them.
    """
    assert all_columns is None, "all_columns!=None is not supported yet"
    fullmatname_col=DDEF().SAMPLE_COLNAME
    assert df_manual[fullmatname_col].equals(df_extra[fullmatname_col])

    cols_checked=[]
    combined=df_manual.copy()
    for col in df_extra.columns:
        if col==fullmatname_col: continue
        if (all_columns is None) or (col in all_columns):
            if (col not in df_manual.columns):
                #logger.debug(f"Adding   column {('\''+col+'\''):30s} from extra split data with no checks")
                combined[col]=df_extra[col]
            else:
                logger.debug(f"Checking column {('\''+col+'\''):30s}")
                comparable_mask=df_manual[col].notna() & df_extra[col].notna()
                if not df_manual.loc[comparable_mask,col].equals(df_extra.loc[comparable_mask,col]):
                    logger.debug(f"dtypes: {(df_manual[col].dtype, df_extra[col].dtype)}")
                    logger.warning(pd.concat([df_manual[col], df_extra[col]], axis=1))
                    assert False, f"Column {col} in {flow_name} does not match extra split data"
                cols_checked.append(col)
        else:
            pass #logger.debug(    f"Ignoring column {('\''+col+'\''):30s} from extra split data is not in analysis columns, ignoring")
    for col in df_manual.columns:
        if col not in cols_checked:
            pass#logger.debug(    f"Adding   column {('\''+col+'\''):30s} from manual split data with no checks")
    if all_columns is not None:
        return combined[all_columns]
    else:
        return combined

class FolderPlusExtraSampleSplitManager(FolderSampleSplitManager):
    def get_complementary_split_table_from_extra(self, df_manual: pd.DataFrame, flow_name: str) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented in a subclass")
    def get_complementary_split_table_columns_from_extra(self, flow_name: str) -> list[DVColumn]:
        raise NotImplementedError("This method should be implemented in a subclass")
    def get_split_table_columns_from_external(self, flow_name: str) -> list[DVColumn]:
        fcols = super().get_split_table_columns_from_external(flow_name)
        ecols=self.get_complementary_split_table_columns_from_extra(flow_name)
        all_columns= []
        for col in fcols + ecols:
            if col.name not in all_columns:
                all_columns.append(col)
        return all_columns
    def get_split_table_from_external(self, flow_name: str) -> pd.DataFrame:
        df_manual = super().get_split_table_from_external(flow_name)
        df_extra = self.get_complementary_split_table_from_extra(df_manual, flow_name)
        all_columns = self.get_split_table_columns_from_external(flow_name)
        combined = check_against_extra_source_and_combine(df_manual, df_extra,
                      all_columns=None,#[DDEF().SAMPLE_COLNAME,*[c.name for c in all_columns]],
                      flow_name=flow_name)
        return combined



        

        
    