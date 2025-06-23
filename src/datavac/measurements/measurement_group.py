from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from datavac.io.measurement_table import MultiUniformMeasurementTable
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

    required_dependencies: Mapping[str,str] = field(default_factory=dict)
    """Mapping of required dependencies groups to their key in the extract_by_umt() kwargs"""

    optional_dependencies: Mapping[str,str] = field(default_factory=dict)
    """Mapping of optional dependencies groups to their key in the extract_by_umt() kwargs"""

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

    def extract_by_mumt(self, measurements: MultiUniformMeasurementTable, **kwargs) -> None:
        """Runs extraction on the measurements.

        Input is a MultiUniformMeasurementTable, which contains multiple UniformMeasurementTables (UMTs).
        Default behavior is to just run self.extract_by_umt on each of UMT individually.  Since UMTs have
        much nicer guarantees on the data for performant extraction (e.g. headers will be regular numpy arrays,
        no extra empty headers from unifying multiple UMTs with different headers), it's generally much easier
        to override extract_by_umt and not this method!

        However, some extractions may require access to the full MultiUniformMeasurementTable all at once
        (e.g. reliability extractions may pull out *deltas* between the first sweep and subsequent ones, 
        so if the reader function results in those sweeps being in different UMTs, it may be necessary to 
        override this method instead).  In that case, a typical sub-class implementation may look like:

        >>>        def extract_by_mumt(self, measurements: UniformMeasurementTable, **kwargs) -> None:
        >>>            # Do something simple with the full MultiUniformMeasurementTable
        >>>            ...
        >>>
        >>>            # Then call the super method to run performant extraction at UMT level
        >>>            super().extract_by_mumt(measurements, **kwargs)
        >>>
        >>>            # Do something simple with the full MultiUniformMeasurementTable again
        >>>            ...
        >>>
        >>>        def extract_by_umt(self, measurements: UniformMeasurementTable, **kwargs) -> None:
        >>>            # Do the actual extraction on the UMT
        >>>            ...

        Note: if you override this method in such a way that it does not call extract_by_umt, then
        extract_by_umt will not be called at all.

        Args:
            measurements: The measurements to extract from.
            **kwargs: Additional MultiUniformMeasurementTables corresponding to the dependencies
        """
        for umt in measurements._umts:
            self.extract_by_umt(umt, **kwargs)

    def extract_by_umt(self, measurements: UniformMeasurementTable, **kwargs) -> None:
        """Runs extraction on the measurements, supplied as a UniformMeasurementTable.

        If possible, it's generally preferable to override this function instead of extract_by_mumt,
        since operations on a UMT are much simpler and more performant.  See extract_by_mumt for
        more details.

        Args:
            measurements: The measurements to extract from.
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
