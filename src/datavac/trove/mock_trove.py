from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator, Union, cast
from datavac.config.data_definition import DVColumn
from datavac.trove import ReaderCard, Trove
if TYPE_CHECKING:
    from datavac.io.measurement_table import MultiUniformMeasurementTable
    import pandas as pd


class MockTrove(Trove):
    load_info_columns:list[DVColumn] = []

    def iter_read(self,
             only_meas_groups: list[str] = None, # type: ignore
             only_sampleload_info: dict[str, list[str]] = {},
             info_already_known: dict = {}, **kwargs)\
                -> Generator[tuple[str,dict[str,dict[str,'MultiUniformMeasurementTable']],
                                        dict[str,dict[str,str]],
                                        dict[str,dict[str,Any]]]]:
        from datavac.config.project_config import PCONF
        from datavac.io.measurement_table import MultiUniformMeasurementTable
        completer=PCONF().data_definition.sample_info_completer
        sample_name_col = PCONF().data_definition.sample_identifier_column.name
        sample_to_mg_to_data = {}
        sample_to_sampleload_info = {}
        for mg_name, mg in PCONF().data_definition.measurement_groups.items():
            if only_meas_groups is not None and mg_name not in only_meas_groups:
                continue
            for rc in mg.reader_cards[self.name]:
                assert isinstance(rc, MockReaderCard), \
                    f"MockTrove only supports MockReaderCard, got {type(rc)}"
                rc=cast(MockReaderCard, rc)
                for sample, sample_info in rc.sample_func().items():
                    sample_info=completer(dict(**sample_info, **info_already_known))
                    if only_sampleload_info is not None:
                        if not all(sample_info[k] in v for k, v in only_sampleload_info.items()):
                            continue
                    if sample not in sample_to_mg_to_data:
                        sample_to_mg_to_data[sample] = {}
                    sample_to_mg_to_data[sample][mg_name]=\
                        MultiUniformMeasurementTable.from_read_data(rc.read(
                            mg_name=mg_name,only_sampleload_info={k:[v] for k,v in sample_info.items()}),mg)
                    sample_to_sampleload_info[sample] = sample_info
        yield '', sample_to_mg_to_data, sample_to_sampleload_info,{}        


@dataclass
class MockReaderCard(ReaderCard):
    reader_func: Callable[...,Union[list[pd.DataFrame], dict[str, list[pd.DataFrame]] ]]= (lambda: [])
    """Implementation of the mock read.
    
    Takes an argument by the name of DDEF().sample_identifier_column.name and other **kwargs,
    returns a suitable list of DataFrames or a dict of lists of DataFrames"""

    sample_func: Callable[[],dict[Any,dict[str,Any]]] = dict
    """Returns a dictionary of sample names to dictionaries of sample information."""

    def read(self, mg_name: str, only_sampleload_info: dict[str, Any] = {}, read_info_so_far: dict[str, Any] = {}, **kwargs)\
            -> Union[list[pd.DataFrame], dict[str, list[pd.DataFrame]]]:
        """Mock read implementation."""
        from datavac.config.data_definition import DDEF
        SAMPLECOLNAME = DDEF().SAMPLE_COLNAME
        assert SAMPLECOLNAME in only_sampleload_info
        rets=[]
        for sn in only_sampleload_info[SAMPLECOLNAME]:
            rets.append(self.reader_func(**{SAMPLECOLNAME: sn,},mg_name=mg_name,**kwargs))
        if not len(rets):
            return []
        if isinstance(rets[0], dict):
            return {k:v for ret in rets for k,v in ret.items()}
        else:
            return sum(rets, start=[])