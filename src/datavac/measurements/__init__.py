from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional


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

    extr_columns: list[DVColumn]
    """The columns that are extracted, i.e. added by extract()."""

    required_dependencies: list[str] = field(default_factory=list)
    """List of required dependencies (names of other measurement groups) for this measurement group."""

    optional_dependencies: list[str] = field(default_factory=list)
    """List of optional dependencies (names of other measurement groups) for this measurement group."""

    subsample_reference_names: list[str] = field(default_factory=list)
    """List of names of subsample references that provide additional information about the measurements."""

    reader_cards: dict[str,list[ReaderCard]] = field(default_factory=dict)
    """Dictionary of reader cards by Trove name that provide information on how to read the measurements.
    For now, all reader cards must belong to the same trove, ie len(self.reader_cards.keys()) == 1.
    """

    def __post_init__(self):
        assert len(self.reader_cards)<=1, \
            "All reader cards in a measurement group must belong to the same trove."     

    def extract(self, measurements: UniformMeasurementTable, **kwargs) -> None:
        """Runs extraction on the measurements
        Args:
            measurements (UniformMeasurementTable): The measurements to extract from.
            **kwargs: Additional UniformMeasurementTables corresponding to the dependencies
        """
        pass

@dataclass
class SemiDevMeasurementGroup(MeasurementGroup):
    layout_param_group: Optional[str] = None
    
    def subsample_reference_names(self) -> list[str]:
        """Returns the subsample references for this measurement group."""
        ssrs = ["Dies"]
        if self.layout_param_group:
            ssrs.append(self.layout_param_group)
        return ssrs
