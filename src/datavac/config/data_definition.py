from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Callable

from datavac.util.lazydict import FunctionLazyDict


if TYPE_CHECKING:
    from numpy import dtype
    import pandas as pd
    from sqlalchemy.sql.sqltypes import TypeEngine
    from datavac.trove import ReaderCard, Trove
    from datavac.io.measurement_table import UniformMeasurementTable, MeasurementTable
    from datavac.measurements.measurement_group import MeasurementGroup


class ModelingType(Enum):
    """Enum for modeling types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    
@dataclass
class DVColumn:
    name: str
    """The name of the column."""

    pd_dtype: str
    """The pandas dtype as a string, e.g. 'float64', 'int32', 'string'."""

    description: str
    """A description of the column."""

    default_modeling_type: Optional[ModelingType] = None
    """The default modeling type for this column"""

    @property
    def sql_dtype(self) -> TypeEngine:
        """The SQLAlchemy type"""
        from datavac.database.db_structure import pd_to_sql_types
        return pd_to_sql_types[self.pd_dtype]
    
@dataclass
class SampleReference():
    """ Sample references connect samples to further information by a reference from the Samples table to another table.
    
    For example, in semiconductor measurements, sample references represent e.g. the mask sets or overall process flow.
    """

    name: str
    """The name of the sample reference."""

    description: str
    """A description of the sample reference."""

    key_column: DVColumn
    """The column that uniquely identifies the sample reference."""

    info_columns: list[DVColumn] = field(default_factory=list)
    """List of columns that provide additional information about the sample reference."""

@dataclass
class SampleDescriptor():
    """ Sample descriptors provide additional information about the samples by another table referencing to the Samples table.

    For example, in semiconductor measurements, sample descriptors might be flow split experiments.  Not every sample is in a
    given split table, so the foreign key goes from the sample descriptor table to the Samples table.
    """

    name: str
    """The name of the sample reference."""

    description: str
    """A description of the sample reference."""

    info_columns: list[DVColumn] = field(default_factory=list)
    """List of columns that provide additional information about the sample reference."""

@dataclass
class SubSampleReference():
    """ Subsample references connect measurements to structural information.
    
    For example, in semiconductor measurements, subsample references represent e.g. tables of dies or layout parameters.
    """
    
    name: str
    """The name of the subsample reference."""

    description: str
    """A description of the subsample reference."""

    key_column: DVColumn
    """The column that uniquely identifies the subsample reference."""
    
    info_columns: list[DVColumn] = field(default_factory=list)
    """List of columns that provide additional information about the subsample reference."""

@dataclass
class HigherAnalysis():
    name: str
    description: str
    analysis_function: Callable[[MeasurementTable], pd.DataFrame]
    required_dependencies: list[str] = field(default_factory=list)
    optional_dependencies: list[str] = field(default_factory=list)

@dataclass
class DataDefinition():

    measurement_groups: Mapping[str, MeasurementGroup] = field(default_factory=dict)
    """ Mapping of measurement groups by name (often dict or CategorizedLazyDict)."""

    higher_analyses: Mapping[str, HigherAnalysis] = field(default_factory=dict)
    """ Mapping of higher analyses by name (often dict or CategorizedLazyDict)."""

    sample_identifier_column: DVColumn = field(default_factory=lambda:\
               DVColumn('SampleName', 'string', 'Unique identifier for each sample'))
    """ Column that uniquely identifies each sample."""

    sample_info_columns: list[DVColumn] = field(default_factory=list)
    """ List of sample info columns (ie potentially non-unique sample information)."""

    sample_info_completer: Callable[[dict[str, Any]], dict[str, Any]] = lambda x: x
    """ A function that takes a dictionary of sample information and returns a completed dictionary."""

    sample_references: Mapping[str,SampleReference] = field(default_factory=dict)
    """ Mapping of sample references by name (often dict or CategorizedLazyDict)."""
    
    sample_descriptors: Mapping[str,SampleDescriptor] = field(default_factory=dict)
    """ Mapping of sample descriptors by name. (often dict or CategorizedLazyDict)."""

    subsample_references: Mapping[str,SubSampleReference] = field(default_factory=dict)
    """ Mapping of subsample references by name (often dict or CategorizedLazyDict)."""

    troves: dict[str, Trove] = None # type: ignore
    """ Dictionary of Troves by name. If not provided, defaults to a single ClassicFolderTrove named 'all'."""

    def __post_init__(self):
        if self.troves is None:
            from datavac.trove.classic_folder_trove import ClassicFolderTrove
            self.troves = {'all': ClassicFolderTrove()}


@dataclass(kw_only=True)
class SemiDeviceDataDefinition(DataDefinition):

    layout_params_dir: Path 
    layout_params_yaml: Path

    sample_references: dict[str,SampleReference]\
                    = field(default_factory=lambda: {'MaskSet':
                          SampleReference(
                            name='MaskSet',
                            description='The mask set used for the sample.',
                            key_column=DVColumn('MaskSet', 'string', 'The mask set used for the sample.'),
                            info_columns=[DVColumn('info_pickle', 'object', 'MaskSet geometric info')])
                        },init=False)
    
    def __post_init__(self):
        super().__post_init__()
        self.sample_descriptors=FunctionLazyDict(
            getter=lambda name: SampleDescriptor(
                name=name,
                description=f'Sample descriptor for flow {name.split(maxsplit=1)[1]}.',
                info_columns=self.get_split_table_columns(name.split(maxsplit=1)[1])),
            keylister=lambda: [f'SplitTable {f}' for f in self.get_flow_names()])
        self.subsample_references = FunctionLazyDict(
            getter=lambda name: self._subsample_reference(name),
            keylister=lambda: self._subsample_reference_names())
            
    def get_layout_params_table(self, lp_group) -> pd.DataFrame:
        """Returns the layout parameters for the given layout parameter group."""
        # TODO: Don't need to load all of them here
        from datavac.io.layout_params import get_layout_params
        return get_layout_params()._tables_by_meas[lp_group]
    def get_layout_params_table_names(self) -> list[str]:
        """Returns a list of all layout parameter group names."""
        # TODO: Don't need to load all of them here
        from datavac.io.layout_params import get_layout_params
        return list(get_layout_params()._tables_by_meas.keys())
    
    def get_split_table_columns(self, flow_name: str) -> list[DVColumn]:
        """Returns the columns of the split table for the given flow name."""
        raise NotImplementedError("This method should be implemented in subclasses to return the split table columns.")
    def get_flow_names(self) -> list[str]:
        """Returns a list of all flow names (each flow name will correspond to a split table)."""
        raise NotImplementedError("This method should be implemented in subclasses to return the flow names.")
    
    def _subsample_reference(self, name: str) -> SubSampleReference:
        if name == 'Dies':
            return SubSampleReference(
                name='Dies',
                description='A subsample reference for dies in a semiconductor sample.',
                key_column=DVColumn('dieid', 'string', 'The unique identifier for each die.'),
                info_columns=[DVColumn('DieX', 'int32', 'The X coordinate of the die.'),
                              DVColumn('DieY', 'int32','The Y coordinate of the die.'),
                              DVColumn('DieRadius [mm]', 'float64', 'The radius of the die from wafer center in mm'),
                              DVColumn('DieComplete', 'boolean', 'Whether the die is complete or partial')])
        elif 'LayoutParams -- ' in name:
            lpg=name.split(" -- ")[-1]
            lpt=self.get_layout_params_table(lpg)
            return SubSampleReference(
                name=name,
                description=f'Layout parameters for "{lpg}".',
                key_column=DVColumn('Structure', 'string', 'The structure measured'),
                info_columns=[DVColumn(c,lpt[c].dtype.name,description=c) for c in lpt.columns
                              if c not in ['Structure','Site']])
        else:
            raise ValueError(f"Subsample reference '{name}' not found in data definition.")
    def _subsample_reference_names(self) -> list[str]:
        """Returns a list of all subsample reference names."""
        return ['Dies'] + [f'LayoutParams -- {lp_group}' for lp_group in self.get_layout_params_table_names()]