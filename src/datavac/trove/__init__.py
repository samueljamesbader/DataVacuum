from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping, Optional, Sequence, Union
from dataclasses import dataclass, field

if TYPE_CHECKING:
    import pandas as pd
    from datavac.config.data_definition import DVColumn
    from datavac.io.measurement_table import MultiUniformMeasurementTable
    from sqlalchemy import Table, MetaData, Connection

@dataclass
class ReaderCard():
    reader_func: Optional[Callable[...,list[pd.DataFrame]|dict[str,list[pd.DataFrame]]]] = None
    """Signature should match self.read()"""

    def read(self, mg_name:str=None, # type: ignore
             only_sampleload_info:dict[str,Any]={}, read_info_so_far:dict[str,Any]={}, **kwargs)\
        -> Union[list[pd.DataFrame], dict[str, list[pd.DataFrame]]]:

        assert mg_name is not None, "mg_name must be specified for ReaderCard.read()"
        if self.reader_func is not None:
            return self.reader_func(mg_name=mg_name, only_sampleload_info=only_sampleload_info.copy(),
                                    read_info_so_far=read_info_so_far.copy(), **kwargs)
        else:
            raise NotImplementedError("Either implement ReaderCard.read() in a subclass or supply reader_func.")

@dataclass
class Trove(): 
    """A source of data to read into the database, e.g. a folder structure or other source database."""

    name: str = ''
    """The name of the trove, used to identify it in the database and in the configuration."""

    load_info_columns: list[DVColumn] = field(default_factory=list)
    """List of columns associated to a load from this trove."""
    
    natural_grouping: Optional[str] = None
    """Name of a sample/load information column that it makes sense to read in altogether.
    
    eg if all data for a given 'Lot' is accessed together, the Trove might use this column to reduce requests."""

    cli_expander: dict[str,dict[str,Any]] = field(default_factory=dict)

    def read(self,*args,**kwargs)\
                 -> tuple[dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]]]:
        total_mg_to_sample_to_data = {}
        total_sample_to_sampleloadinfo = {}
        for readgrp_name, mg_to_sample_to_data, sample_to_sampleloadinfo,_ in self.iter_read(*args,**kwargs):
            for mg_name, sample_to_data in mg_to_sample_to_data.items():
                if mg_name not in total_mg_to_sample_to_data:
                    total_mg_to_sample_to_data[mg_name] = {}
                for sample_name, data in sample_to_data.items():
                    assert sample_name not in total_mg_to_sample_to_data[mg_name], f"Same sample name '{sample_name}' encountered in two different reads."
                    total_mg_to_sample_to_data[mg_name][sample_name] = data
            for sample_name, sampleloadinfo in sample_to_sampleloadinfo.items():
                assert sample_name not in total_sample_to_sampleloadinfo, f"Same sample name '{sample_name}' encountered in two different reads."
                total_sample_to_sampleloadinfo[sample_name] = sampleloadinfo
        return total_mg_to_sample_to_data, total_sample_to_sampleloadinfo

    def iter_read(self,
             only_meas_groups:Optional[list[str]]=None,
             only_sampleload_info:dict[str,Sequence[Any]]={},
             info_already_known:dict={}, incremental:bool=False, **kwargs)\
                 -> Generator[tuple[str,dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]],dict[str,dict[str,Any]]]]:
        """Reads data from the trove.

        Args:
            only_meas_groups: If set, only reads the measurement groups with these names.
            only_sampleload_info: If set, restricts to records which match these information
                Each key is a column name in the DataDefinition sample info or Trove load info,
                and each value is a sequence of values that are allowed for that column.
            info_already_known: If set, this is a dictionary of information that is already known
                about the sample or load based on context.
            incremental: If True, this read is being done as part of an incremental upload,
                and the trove can optimize for this if desired.
            **kwargs: Additional arguments that may be used by the specific Trove implementation.
        Yields:
            A tuple of a string and two dictionaries:
            - The string is an arbitrary identifier for the grouping of data that was read with no guarantees
            - The first dictionary maps measurement group names to dictionaries of sample names to
              MultiUniformMeasurementTables.
            - The second dictionary maps sample names to dictionaries of sample/load information.
        """
        raise NotImplementedError("Trove.iter_read() must be implemented by subclasses.")
    
    def get_natural_grouping_factors(self) -> list[Any]:
        """Returns a list of factor values that can be used to narrow down the read_dir based on the natural grouping."""
        raise NotImplementedError("Trove.get_natural_grouping_factors() must be implemented by subclasses.")
    
    def dbtables(self,key: str) -> Table:
        """Returns the database table associated with this trove for the given key."""
        from datavac.database.db_structure import DBSTRUCT
        return DBSTRUCT().get_trove_dbtables(self.name)[key]

    def additional_tables(self, int_schema: str, metadata: 'MetaData', load_tab: 'Table') -> tuple[dict[str, Table], dict[str, Table]]:
        """Returns additional database tables that should be created for this trove,
        e.g. for additional information associated with the load.
        
        Note: when implementing, see datavac.database.db_structure.DBSTRUCT.get_trove_dbtables()
        for how to integrate these additional tables into the database structure and follow the pattern.
        
        The first return is for data tables that should be included in getting data, while the second are for trove's internal use.
        """
        return {},{}
    
    def trove_reference_columns(self) -> Mapping[DVColumn,str]:
        """Returns a mapping of DVColumns that are associated with this trove to the relevant trove table name"""
        return {}
    
    def transform(self, df: pd.DataFrame, loadid: int, sample_info:dict[str,Any], conn:Optional[Connection]=None, readgrp_name:str|None=None):
        """Applies a transformation to the dataframe to e.g. map read info to trove table columns.
        
        May modify the trove tables, but will not commit the connection (if the connection is provided)."""
        pass

    def affected_meas_groups_and_filters(self, samplename: Any, comp: dict[str,dict[str,Any]],
                                         data_by_mg: dict[str,MultiUniformMeasurementTable], conn: Optional[Connection] = None)\
            -> tuple[list[str], list[str], dict[str,Sequence[Any]]]:
        raise NotImplementedError("Trove.affected_meas_groups() must be implemented by subclasses if incremental upload is desired.")