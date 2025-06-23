from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union
from dataclasses import dataclass, field



if TYPE_CHECKING:
    import pandas as pd
    from datavac.config.data_definition import DVColumn
    from datavac.io.measurement_table import MultiUniformMeasurementTable

@dataclass
class ReaderCard():
    reader_func: Callable[...,list[pd.DataFrame]|dict[str,list[pd.DataFrame]]]

@dataclass
class Trove(): 
    """A source of data to read into the database, e.g. a folder structure or other source database."""

    name: str = ''
    """The name of the trove, used to identify it in the database and in the configuration."""

    load_info_columns: list[DVColumn] = field(default_factory=list)
    """List of columns associated to a load from this trove."""

    def read(self,
             only_meas_groups:Optional[list[str]]=None,
             only_sampleload_info:dict[str,Sequence[Any]]={},
             info_already_known:dict={}, **kwargs)\
                 -> tuple[dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]]]:
        """Reads data from the trove.

        Args:
            only_meas_groups: If set, only reads the measurement groups with these names.
            only_sampleload_info: If set, restricts to records which match these information
                Each key is a column name in the DataDefinition sample info or Trove load info,
                and each value is a sequence of values that are allowed for that column.
            info_already_known: If set, this is a dictionary of information that is already known
                about the sample or load based on context.
            **kwargs: Additional arguments that may be used by the specific Trove implementation.
        Returns:
            A tuple of two dictionaries:
            - The first dictionary maps measurement group names to dictionaries of sample names to
              MultiUniformMeasurementTables.
            - The second dictionary maps sample names to dictionaries of sample/load information.
        """
        raise NotImplementedError("Trove.read() must be implemented by subclasses.")

