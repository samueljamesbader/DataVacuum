from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from datavac.util.util import only


if TYPE_CHECKING:
    from datavac.config.data_definition import DVColumn, SubSampleReference
    from datavac.io.measurement_table import UniformMeasurementTable
    from datavac.trove import ReaderCard

@dataclass
class MeasurementGroup():
    name: str
    """The name of the measurement group."""

    description: str
    """A description of the measurement group."""

    meas_columns: list[DVColumn]
    """The measurement columns that are read in for this measurement group."""

    extr_column_names: list[str]
    """The columns names that are extracted and kept.
    
    Must be a subset of names in get_available_extr_columns().
    """

    required_dependencies: list[str] = field(default_factory=list)
    """List of required dependencies (names of other measurement groups) for this measurement group."""

    optional_dependencies: list[str] = field(default_factory=list)
    """List of optional dependencies (names of other measurement groups) for this measurement group."""

    subsample_reference_names: list[str] = field(default_factory=list)
    """List of names of subsample references that provide additional information about the measurements."""

    reader_cards: Mapping[str,Sequence[ReaderCard]] = field(default_factory=dict)
    """Dictionary of reader cards by Trove name that provide information on how to read the measurements.
    For now, all reader cards must belong to the same trove, ie len(self.reader_cards.keys()) == 1.
    """

    def __post_init__(self):
        assert len(self.reader_cards)<=1, \
            "All reader cards in a measurement group must belong to the same trove."     

    def available_extr_columns(self) -> dict[str,DVColumn]:
        """Returns the columns that are available for extraction in this measurement group."""
        return {}

    def extract(self, measurements: UniformMeasurementTable, **kwargs) -> None:
        """Runs extraction on the measurements
        Args:
            measurements (UniformMeasurementTable): The measurements to extract from.
            **kwargs: Additional UniformMeasurementTables corresponding to the dependencies
        """
        pass

    def trove_name(self) -> str:
        """Returns the name of the trove that contains the reader cards for this measurement group."""
        if not len(self.reader_cards): return ''
        return only(self.reader_cards.keys())

@dataclass
class SemiDevMeasurementGroup(MeasurementGroup):
    layout_param_group: Optional[str] = None
    
    subsample_reference_names: list[str]= field(init=False,
            default=property(lambda self: self._subsample_reference_names())) # type: ignore
    def _subsample_reference_names(self) -> list[str]:
        """Returns the subsample references for this measurement group."""
        ssrs = ["Dies"]
        if self.layout_param_group:
            ssrs.append(f'LayoutParams -- {self.layout_param_group}')
        return ssrs
