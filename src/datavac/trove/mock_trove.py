from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, cast
from datavac.config.data_definition import DVColumn
from datavac.trove import ReaderCard, Trove
if TYPE_CHECKING:
    from datavac.io.measurement_table import MultiUniformMeasurementTable


class MockTrove(Trove):
    load_info_columns:list[DVColumn] = []

    def read(self,
             only_meas_groups: list[str] = None,
             only_sampleload_info: dict[str, list[str]] = {},
             info_already_known: dict = {}, **kwargs)\
            -> tuple[dict[str, dict[str, MultiUniformMeasurementTable]], dict[str, dict[str, str]]]:
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
        return sample_to_mg_to_data, sample_to_sampleload_info           


@dataclass
class MockReaderCard(ReaderCard):
    reader_func: Callable = (lambda: [])
    """Implementation of ReaderCard.read()."""

    sample_func: Callable[[],dict[Any,dict[str,Any]]] = dict
    """Returns a dictionary of sample names to dictionaries of sample information."""