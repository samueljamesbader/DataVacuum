from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Callable


if TYPE_CHECKING:
    from numpy import dtype
    import pandas as pd
    from sqlalchemy.sql.sqltypes import TypeEngine
    from datavac.trove import ReaderCard, Trove
    from datavac.io.measurement_table import UniformMeasurementTable, MeasurementTable
    from datavac.measurements import MeasurementGroup


class ModelingType(Enum):
    """Enum for modeling types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    
@dataclass
class DVColumn:
    name: str
    """The name of the column."""

    description: str
    """A description of the column."""

    pd_dtype: str
    """The pandas dtype as a string, e.g. 'float64', 'int32', 'string'."""

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

    _measurement_groups: dict[str, MeasurementGroup] = field(default_factory=dict)
    """ Dictionary of measurement groups by name."""

    higher_analyses: dict[str, HigherAnalysis] = field(default_factory=dict)
    """ Dictionary of higher analyses by name."""

    sample_identifier_column: DVColumn = field(default_factory=lambda: DVColumn('SampleName', 'Unique identifier for each sample', pd_dtype='string'))
    """ Column that uniquely identifies each sample."""

    sample_info_columns: list[DVColumn] = field(default_factory=list)
    """ List of sample info columns (ie potentially non-unique sample information)."""

    _sample_references: dict[str,SampleReference] = field(default_factory=dict)
    """ Dictionary of sample references by name."""
    
    _sample_descriptors: dict[str,SampleDescriptor] = field(default_factory=dict)
    """ Dictionary of sample descriptors by name."""

    _subsample_references: dict[str,SubSampleReference] = field(default_factory=dict)
    """ Dictionary of subsample references by name."""

    _troves: dict[str, Trove] = None # type: ignore
    """ Dictionary of Troves by name. If not provided, defaults to a single ClassicFolderTrove named 'all'."""

    def __post_init__(self):
        if self.troves is None:
            from datavac.trove.classic_folder_trove import ClassicFolderTrove
            self.troves = {'all': ClassicFolderTrove()}

    def measurement_group(self, name: str) -> MeasurementGroup:
        """Returns the MeasurementGroup with the given name.
         
        The default behavior is to reference self._measurement_groups, but subclasses can override,
        e.g., if the preference is to lazily generate measurement groups to avoid import times.
        """
        if name not in self._measurement_groups:
            raise ValueError(f"Measurement group '{name}' not found in data definition.")
        return self._measurement_groups[name]
    
    def measurement_group_names(self) -> list[str]:
        """Returns a list of all measurement group names.
        
        The default behavior is to reference self._measurement_groups, but subclasses can override,
        e.g., if the preference is to lazily generate measurement groups to avoid import times.
        """
        return list(self._measurement_groups.keys())
    
    def sample_reference(self, name: str) -> SampleReference:
        """Returns the SampleReference with the given name.
        
        The default behavior is to reference self._sample_references, but subclasses can override,
        e.g., if the preference is to lazily generate sample references to avoid import times.
        """
        if name not in self._sample_references:
            raise ValueError(f"Sample reference '{name}' not found in data definition.")
        return self._sample_references[name]

    def sample_reference_names(self) -> list[str]:
        return ['MaskSet']
    
    def sample_descriptor(self,name: str) -> SampleDescriptor:
        """Returns the SampleDescriptor with the given name.

        The default behavior is to reference self._sample_descriptors, but subclasses can override, 
        e.g., if the preference is to lazily generate sample descriptors to avoid import times.
        """
        if name not in self._sample_descriptors:
            raise ValueError(f"Sample descriptor '{name}' not found in data definition.")
        return self._sample_descriptors[name]

    def sample_descriptor_names(self) -> list[str]:
        """Returns a list of all sample descriptor names.

        The default behavior is to reference self._sample_descriptors, but subclasses can override,
        e.g., if the preference is to lazily generate sample descriptors to avoid import times.
        """
        return list(self._sample_descriptors.keys())

    def subsample_reference(self, name: str) -> SubSampleReference:
        """Returns the SubSampleReference with the given name.

        The default behavior is to reference self._subsample_references, but subclasses can override,
        e.g., if the preference is to lazily generate subsample references to avoid import times.
        """
        if name not in self._subsample_references:
            raise ValueError(f"Subsample reference '{name}' not found in data definition.")
        return self._subsample_references[name]

    def subsample_reference_names(self) -> list[str]:
        """Returns a list of all subsample reference names.

        The default behavior is to reference self._subsample_references, but subclasses can override,
        e.g., if the preference is to lazily generate subsample references to avoid import times."""
        return list(self._subsample_references.keys())

    def trove(self, name: str) -> Trove:
        """Returns the Trove with the given name.
        
        The default behavior is to reference self._troves, but subclasses can override,
        e.g., if the preference is to lazily generate troves to avoid import times.
        """
        if name not in self.troves:
            raise ValueError(f"Trove '{name}' not found in data definition.")
        return self.troves[name]

@dataclass
class SemiDeviceDataDefinition(DataDefinition):

    _sample_references = {'MaskSet':
                          SampleReference(
                            name='MaskSet',
                            description='The mask set used for the sample.',
                            key_column=DVColumn('MaskSet', 'The mask set used for the sample.', pd_dtype='string'),
                            info_columns=[DVColumn('info_pickle', 'MaskSet geometric info', pd_dtype='object')])
                        }
    
    def get_layout_params_table(self, lp_group) -> pd.DataFrame:
        """Returns the layout parameters for the given layout parameter group."""
        from datavac.io.layout_params import get_layout_params
        return get_layout_params()._tables_by_meas[lp_group]
    def get_layout_params_table_names(self) -> list[str]:
        """Returns a list of all layout parameter group names."""
        from datavac.io.layout_params import get_layout_params
        return list(get_layout_params()._tables_by_meas.keys())
    
    def get_split_table_columns(self, flow_name: str) -> list[DVColumn]:
        """Returns the columns of the split table for the given flow name."""
        raise NotImplementedError("This method should be implemented in subclasses to return the split table columns.")
    def get_flow_names(self) -> list[str]:
        """Returns a list of all flow names (each flow name will correspond to a split table)."""
        raise NotImplementedError("This method should be implemented in subclasses to return the flow names.")

    def sample_descriptor(self, name: str) -> SampleDescriptor:
        if 'SplitTable' in name:
            return SampleDescriptor(
                name=name,
                description='A sample descriptor for a flow split experiment.',
                info_columns=self.get_split_table_columns(name))
        else: raise ValueError(f"Sample descriptor '{name}' not found in data definition.")

    def sample_descriptor_names(self) -> list[str]:
        return [f'SplitTable_{flow_name}' for flow_name in self.get_flow_names()]
    
    def subsample_reference(self, name: str) -> SubSampleReference:
        if name == 'Dies':
            return SubSampleReference(
                name='Dies',
                description='A subsample reference for dies in a semiconductor sample.',
                key_column=DVColumn('dieid', 'The unique identifier for each die.', pd_dtype='string'),
                info_columns=[DVColumn('DieX', 'The X coordinate of the die.', pd_dtype='int32'),
                              DVColumn('DieY', 'The Y coordinate of the die.', pd_dtype='int32'),
                              DVColumn('DieRadius [mm]', 'The radius of the die from wafer center in mm', pd_dtype='float64'),
                              DVColumn('DieComplete', 'Whether the die is complete or partial', pd_dtype='boolean')])
        elif 'LayoutParams -- ' in name:
            lpg=name.split(" -- ")[-1]
            lpt=self.get_layout_params_table(lpg)
            return SubSampleReference(
                name=name,
                description=f'Layout parameters for "{lpg}".',
                key_column=DVColumn('Structure', 'The structure measureed', pd_dtype='string'),
                info_columns=[DVColumn(c,c,lpt[c].dtype.name) for c in lpt.columns
                              if c not in ['Structure','Site']])
        else:
            raise ValueError(f"Subsample reference '{name}' not found in data definition.")
    def subsample_reference_names(self) -> list[str]:
        """Returns a list of all subsample reference names."""
        return ['Dies'] + [f'LayoutParams -- {lp_group}' for lp_group in self.get_layout_params_table_names()]