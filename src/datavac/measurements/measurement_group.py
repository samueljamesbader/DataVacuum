from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from datavac.io.measurement_table import UniformMeasurementTable
from datavac.util.util import only


if TYPE_CHECKING:
    from datavac.config.data_definition import DVColumn, SubSampleReference
    from datavac.io.measurement_table import UniformMeasurementTable, MultiUniformMeasurementTable
    from datavac.trove import Trove, ReaderCard
    from sqlalchemy import Table

@dataclass(eq=False,repr=False)
class MeasurementGroup():
    """Represents a collection of measurements that will be stored in one table.

    Notes: there should only be one MeasurementGroup with a given name, so hashing/equality is based
    on the name.  This an implementation detail, but it does mean if you subclass this class and if
    you decorate the subclass with @dataclass, you should set eq=False to avoid dataclass overriding
    the __hash__/__eq__ method.  See https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass.
    If you fail to do so, Python will complain that the instance is not hashable.
    """
    name: str
    """The name of the measurement group."""

    description: str
    """A description of the measurement group."""

    meas_columns: list[DVColumn]
    """The measurement columns that are read in for this measurement group."""

    only_extr_columns: Optional[list[str]] = None
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
    
    involves_sweeps: bool = True
    """Whether this measurement group includes non-scalar data (e.g. swept arrays) that require a BYTEA sweep table"""

    def __post_init__(self):
        assert len(self.reader_cards)<=1, \
            "All reader cards in a measurement group must belong to the same trove."     
    
    @cached_property
    def extr_column_names(self) -> list[str]:
        if self.only_extr_columns is not None:
            return self.only_extr_columns
        else: 
            return list(self.available_extr_columns().keys())

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

        Note: if you override extract_by_mumt in a way that it does not call extract_by_umt, then
        extract_by_umt will not be called.

        Args:
            measurements: The measurements to extract from.
            **kwargs: Additional UniformMeasurementTables corresponding to the dependencies
        """
        raise NotImplementedError(f"extract_by_umt must be implemented in the subclass (called from {self.__class__.__name__})")
        #pass

    def trove_name(self) -> str:
        """Returns the name of the trove that contains the reader cards for this measurement group."""
        if not len(self.reader_cards): return ''
        return only(self.reader_cards.keys())
    def trove(self) -> Trove:
        """Returns the Trove object that contains the reader cards for this measurement group."""
        from datavac.config.data_definition import DDEF
        return DDEF().troves[self.trove_name()]
    
    def dbtable(self, key:str) -> Table:
        """Returns the SQLAlchemy Table object for this measurement group from DBSTRUCT()."""
        from datavac.database.db_structure import DBSTRUCT
        return DBSTRUCT().get_measurement_group_dbtables(self.name)[key]
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        """Equality is based on the name of the analysis."""
        if isinstance(other, str):
            raise Exception("Comparing MeasurementGroup to string")
        if not isinstance(other, MeasurementGroup):
            return False
        return self.name == other.name
    
    def __str__(self) -> str:
        return f'<MeasGroup:"{self.name}">'
    def __repr__(self) -> str:
        return f'<MeasGroup:"{self.name}">'

@dataclass(eq=False)
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

# TODO: Remove
class NoExtrSemiDevMeasurementGroup(SemiDevMeasurementGroup):
    def extract_by_umt(self, measurements: UniformMeasurementTable, **kwargs) -> None: pass



from typing import TypeVar
T=TypeVar('T', bound=MeasurementGroup)
class ExtractionAddon:
    def additional_extract_by_mumt(self, measurements:MultiUniformMeasurementTable, **kwargs):
        for umt in measurements._umts:
            self.additional_extract_by_umt(umt, **kwargs)
    def additional_extract_by_umt(self, measurements:UniformMeasurementTable, **kwargs):
        pass
    def additional_available_extr_columns(self) -> dict[str, DVColumn]:
        return {}
    def apply_to(self, other: T) -> T:
        from copy import copy
        newmg=copy(other)

        @wraps(other.extract_by_umt)
        def extract_by_mumt(measurements: MultiUniformMeasurementTable, **kwargs):
            other.extract_by_mumt(measurements, **kwargs)
            self.additional_extract_by_mumt(measurements, **kwargs)
        @wraps(other.available_extr_columns)
        def available_extr_columns() -> dict[str, DVColumn]:
            return {**other.available_extr_columns(), **self.additional_available_extr_columns()}
        newmg.extract_by_mumt=extract_by_mumt
        newmg.available_extr_columns=available_extr_columns
        return newmg
    def compose_after(self, other: 'ExtractionAddon') -> 'ExtractionAddon':
        """Composes this addon after another one."""
        class ComposedAddon(ExtractionAddon):
            def additional_extract_by_mumt(self, measurements:MultiUniformMeasurementTable, **kwargs):
                other.additional_extract_by_mumt(measurements, **kwargs)
                self.additional_extract_by_mumt(measurements, **kwargs)
            def additional_available_extr_columns(self) -> dict[str, DVColumn]:
                return {**other.additional_available_extr_columns(), **self.additional_available_extr_columns()}
        return ComposedAddon()
    def __radd__(self, other: T) -> T:
        if isinstance(other, MeasurementGroup):
            return self.apply_to(other)
        elif isinstance(other, ExtractionAddon):
            return self.compose_after(other)