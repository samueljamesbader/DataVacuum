from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, cast, Any

from datavac.appserve.api import client_server_split

if TYPE_CHECKING:
    import pandas as pd
    from pandas import DataFrame
    from datavac.config.data_definition import DVColumn
else:
    # Otherwise Pydantic's validate_call won't be able to parse the types
    DataFrame = Any

@dataclass
class SampleSplitManager():

    def get_flow_names_from_external(self) -> list[str]:
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def _get_flow_names_from_database(self) -> list[str]:
        import pandas as pd
        from datavac.database.db_connect import get_engine_ro, can_try_db
        with get_engine_ro().begin() as conn:
            from sqlalchemy import text
            return list(pd.read_sql(text("""
                SELECT substring(table_name FROM 'SplitTable -- (.*)') AS split_name
                FROM information_schema.tables
                WHERE table_schema = 'vac' AND table_name LIKE 'SplitTable -- %';"""),conn)['split_name'])
    _cached_external_flow_names: Optional[list[str]] = None
    _cached_database_flow_names: Optional[list[str]] = None
    def _get_flow_names(self, force_external:bool=False) -> list[str]:
        from datavac.database.db_connect import get_engine_ro, can_try_db
        if not can_try_db(): force_external=True
        if self._cached_external_flow_names:
            return self._cached_external_flow_names
        elif force_external:
            self._cached_external_flow_names=self.get_flow_names_from_external()
            return self._cached_external_flow_names
        else:
            if self._cached_database_flow_names is None:
                self._cached_database_flow_names = self._get_flow_names_from_database()
            return self._cached_database_flow_names

    def get_split_table_from_external(self, flow_name: str) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented in a subclass")    
    def _get_split_table_from_database(self, flow_name: str) -> pd.DataFrame:
        import pandas as pd
        from datavac.config.data_definition import DDEF
        from datavac.database.db_connect import get_engine_ro, can_try_db
        with get_engine_ro().begin() as conn:
            return pd.read_sql(f"""SELECT s."{DDEF().SAMPLE_COLNAME}", sp.*
                               FROM vac.\"SplitTable -- {flow_name}\" sp
                               JOIN vac.\"Samples\" s on s.sampleid=sp.sampleid""", conn).drop(columns=['sampleid'])
    _cached_external_split_tables: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    _cached_database_split_tables: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    def _get_split_table(self, flow_name: str, force_external:bool=False) -> pd.DataFrame:
        from datavac.database.db_connect import get_engine_ro, can_try_db
        if not can_try_db(): force_external=True
        if self._cached_external_split_tables.get(flow_name) is not None:
            return self._cached_external_split_tables[flow_name]
        elif force_external:
            self._cached_external_split_tables[flow_name] = self.get_split_table_from_external(flow_name)
            return self._cached_external_split_tables[flow_name]
        else:
            if self._cached_database_split_tables.get(flow_name) is None:
                self._cached_database_split_tables[flow_name] = self._get_split_table_from_database(flow_name)
            return self._cached_database_split_tables[flow_name]

    def get_split_table_columns_from_external(self, flow_name: str) -> list[DVColumn]:
        raise NotImplementedError("This method should be implemented in a subclass")
    def _get_split_table_columns_from_database(self, flow_name: str) -> list[DVColumn]:
        from datavac.config.data_definition import DDEF
        split_table = self._get_split_table(flow_name)
        from datavac.config.data_definition import DVColumn
        return [DVColumn(name=col, pd_dtype=str(split_table[col].dtype), description=col)
                for col in split_table.columns if col != DDEF().SAMPLE_COLNAME]
    _cached_external_split_table_columns: dict[str, list[DVColumn]] = field(default_factory=dict, init=False)
    _cached_database_split_table_columns: dict[str, list[DVColumn]] = field(default_factory=dict, init=False)
    def get_split_table_columns(self, flow_name: str, force_external: bool=False) -> list[DVColumn]:
        from datavac.database.db_connect import get_engine_ro, can_try_db
        if not can_try_db(): force_external=True
        if self._cached_external_split_table_columns.get(flow_name) is not None:
            return self._cached_external_split_table_columns[flow_name]
        elif force_external:
            self._cached_external_split_table_columns[flow_name] = self.get_split_table_columns_from_external(flow_name)
            return self._cached_external_split_table_columns[flow_name]
        else:
            if self._cached_database_split_table_columns.get(flow_name) is None:
                self._cached_database_split_table_columns[flow_name] = \
                    self._get_split_table_columns_from_database(flow_name)
            return self._cached_database_split_table_columns[flow_name]

@dataclass(kw_only=True)
class DictSampleSplitManager(SampleSplitManager):

    split_tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get_flow_names_from_external(self) -> list[str]:
        return list(self.split_tables.keys())

    def get_split_table_from_external(self, flow_name: str) -> pd.DataFrame:
        if flow_name not in self.split_tables:
            raise ValueError(f"Flow name '{flow_name}' not found in provided split tables.")
        return self.split_tables[flow_name]
    
    def get_split_table_columns_from_external(self, flow_name: str) -> list[DVColumn]:
        split_table = self.get_split_table_from_external(flow_name)
        from datavac.config.data_definition import DVColumn, DDEF
        return [DVColumn(name=col, pd_dtype=str(split_table[col].dtype),
                         description=col) for col in split_table.columns
                     if col!=DDEF().SAMPLE_COLNAME]
        
@client_server_split('get_flow_names', return_type='ast', split_on='is_server')
def get_flow_names(*args,**kwargs) -> list[str]:
    from datavac.config.data_definition import DDEF, SemiDeviceDataDefinition
    return cast(SemiDeviceDataDefinition,DDEF()).split_manager._get_flow_names(*args,**kwargs)

@client_server_split('get_split_table', return_type='pd', split_on='is_server')
def get_split_table(*args,**kwargs) -> DataFrame:
    from datavac.config.data_definition import DDEF, SemiDeviceDataDefinition
    return cast(SemiDeviceDataDefinition,DDEF()).split_manager._get_split_table(*args,**kwargs)
