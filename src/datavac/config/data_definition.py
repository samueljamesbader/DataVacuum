from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from enum import Enum
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Callable, Sequence, Tuple

from datavac.config.sample_splits import DictSampleSplitManager, SampleSplitManager
from datavac.util.caching import cache_but_copy
from datavac.util.lazydict import FunctionLazyDict
from datavac.util.util import only


if TYPE_CHECKING:
    from numpy import dtype
    import pandas as pd
    from pandas import DataFrame
    from sqlalchemy.sql.sqltypes import TypeEngine
    from sqlalchemy import Constraint, Connection, Table
    from datavac.trove import ReaderCard, Trove
    from datavac.io.measurement_table import UniformMeasurementTable, MeasurementTable
    from datavac.measurements.measurement_group import MeasurementGroup
    from datavac.config.layout_params import LayoutParameters


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
    
    def dbtable(self) -> Table:
        """Returns the database table associated with this subsample reference."""
        from datavac.database.db_structure import DBSTRUCT
        return DBSTRUCT().get_sample_reference_dbtable(self.name)

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

    def get_constraints(self) -> Sequence[Constraint]:
        """Returns a sequence of SQLAlchemy constraints for this subsample reference."""
        return []
    
    def transform(self, table: pd.DataFrame, sample_info: dict[str,Any], conn: Connection) -> pd.DataFrame:
        """Transforms the given table to include the subsample reference information."""
        # Default implementation does nothing, can be overridden in subclasses
        return table
    
    def dbtable(self) -> Table:
        """Returns the database table associated with this subsample reference."""
        from datavac.database.db_structure import DBSTRUCT
        return DBSTRUCT().get_subsample_reference_dbtable(self.name)

@dataclass(eq=False,kw_only=True)
class HigherAnalysis():
    name: str
    description: str
    analysis_function: Optional[Callable[..., pd.DataFrame]] = None
    """The function that performs the analysis (alternative to subclassing and overriding analyze())."""

    only_anls_columns: Optional[list[str]] = None
    
    required_dependencies: Mapping[str,str] = field(default_factory=dict)
    """Mapping of required dependencies by name (MeasurementGroup or HigherAnalysis).
    
    The keys are the names of the MeasurementGroups or HigherAnalyses,
    and the values are the kwarg names with which each will be provided to the analysis function."""
    optional_dependencies: Mapping[str,str] = field(default_factory=dict)
    """Mapping of optional dependencies by name (MeasurementGroup or HigherAnalysis).
    
    The keys are the names of the MeasurementGroups or HigherAnalyses,
    and the values are the kwarg names with which each will be provided to the analysis function"""

    subsample_reference_names: list[str] = field(default_factory=list)

    @cached_property
    def analysis_column_names(self) -> list[str]:
        if self.only_anls_columns is not None:
            return self.only_anls_columns
        else: 
            return list(self.available_analysis_columns().keys())

    def dbtables(self, key:str) -> Table:
        """Returns the database table associated with this higher analysis for the given key."""
        from datavac.database.db_structure import DBSTRUCT
        return DBSTRUCT().get_higher_analysis_dbtables(self.name)[key]
    
    def analyze(self,**kwargs) -> pd.DataFrame:
        """The arguments that analysis_function needs to accept are the required and optional dependencies.
        
        If a dependency is a MeasurementGroup, the function will receive a MultiUniformMeasurementTable.
        If a dependency is a HigherAnalysis, the function will receive a DataFrame.
        If a dependency is optional, the function must not require that argument. 

        The function should return a DataFrame with the analysis results.
        """
        if self.analysis_function is None:
            raise NotImplementedError("HigherAnalysis subclasses must either implement analyze() or provide an analysis_function.")
        else: return self.analysis_function(**kwargs) 

    def available_analysis_columns(self) -> dict[str,DVColumn]:
        """Returns the columns that are available for analysis in this higher analysis."""
        return {}
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __str__(self) -> str:
        return f'<Analysis:"{self.name}">'
    def __repr__(self) -> str:
        return f'<Analysis:"{self.name}">'
    
    def __eq__(self, other: object) -> bool:
        """Equality is based on the name of the analysis."""
        if isinstance(other, str):
            raise Exception("Comparing HigherAnalysis to string")
        if not isinstance(other, HigherAnalysis):
            return False
        return self.name == other.name

@dataclass(eq=False)
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
        for meas_group in self.measurement_groups.values():
            try: hash(meas_group)
            except TypeError:
                raise TypeError(f"MeasurementGroup \"{meas_group.name}\" is not hashable. "\
                                "Usuallly this is because a subclass used @dataclass() instead of @dataclass(eq=False).")
        for higher_analysis in self.higher_analyses.values():
            try: hash(higher_analysis)
            except TypeError:
                raise TypeError(f"HigherAnalysis \"{higher_analysis.name}\" is not hashable. "\
                                "Usuallly this is because a subclass used @dataclass() instead of @dataclass(eq=False).")
    
    @property
    def SAMPLE_COLNAME(self):
        return self.sample_identifier_column.name
    
    @property
    def ALL_SAMPLE_COLNAMES(self):
        return [self.SAMPLE_COLNAME] +\
               [c.name for c in self.sample_info_columns] + \
               [sr.key_column.name for sr in self.sample_references.values()]
    
    def ALL_LOAD_COLNAMES(self, trove_name: str):
        return [c.name for c in self.troves[trove_name].load_info_columns]
    
    def ALL_SAMPLELOAD_COLNAMES(self, trove_name: str):
        return self.ALL_SAMPLE_COLNAMES + self.ALL_LOAD_COLNAMES(trove_name)
    
    @cache
    def get_meas_groups_dependency_graph(self) -> Mapping[MeasurementGroup, Sequence[MeasurementGroup]]:
        """Returns a mapping of measurement groups to their required and optional dependencies."""
        return {mg: [self.measurement_groups[mg_name]
                      for mg_name in list(mg.required_dependencies)\
                                   + list(mg.optional_dependencies)]
                    for mg in self.measurement_groups.values()}
    
    @cache
    def get_analyses_dependency_graph(self) -> Mapping[HigherAnalysis, Sequence[MeasurementGroup|HigherAnalysis]]:
        """Returns a mapping of higher analyses to their required and optional dependencies."""
        return {an: [(self.higher_analyses.get(mgoa_name) or self.measurement_groups[mgoa_name])
                      for mgoa_name in list(an.required_dependencies)\
                                     + list(an.optional_dependencies)]
                    for an in self.higher_analyses.values()}
    @cache
    def get_meas_groups_dependent_graph(self) -> Mapping[MeasurementGroup, Sequence[MeasurementGroup|HigherAnalysis]]:
        """Returns a mapping of measurement groups to their dependents (measurement groups or analyses)."""
        from datavac.util.dag import reverse_dag
        from datavac.measurements.measurement_group import MeasurementGroup
        return reverse_dag({k:[mgoa for mgoa in v if isinstance(mgoa, MeasurementGroup)]
                            for k,v in list(self.get_meas_groups_dependency_graph().items())\
                                          +list(self.get_analyses_dependency_graph().items())})
    @cache
    def get_analyses_dependent_graph(self) -> Mapping[HigherAnalysis, Sequence[HigherAnalysis]]:
        """Returns a mapping of higher analyses to their dependents."""
        from datavac.util.dag import reverse_dag
        return reverse_dag({k:[mgoa for mgoa in v if isinstance(mgoa,HigherAnalysis)]
                            for k,v in self.get_analyses_dependency_graph().items()})

    @cache_but_copy
    def get_meas_groups_topo_sorted(self) -> list[MeasurementGroup]:
        """Returns a list of all meas groups such that any dependencies appear before dependents."""
        from graphlib import TopologicalSorter
        ts = TopologicalSorter(self.get_meas_groups_dependency_graph())
        return list(ts.static_order())
    
    @cache_but_copy
    def get_higher_analyses_topo_sorted(self) -> list[HigherAnalysis]:
        """Returns a list of all higher analyses such that any dependencies appear before dependents."""
        from graphlib import TopologicalSorter
        ts = TopologicalSorter({k:[mgoa for mgoa in v if isinstance(mgoa,HigherAnalysis)]
                                for k,v in self.get_analyses_dependency_graph().items()})
        return list(ts.static_order())
    
    def populate_initial(self, conn: Optional[Connection] = None ): pass
    

@dataclass(kw_only=True,eq=False)
class SemiDeviceDataDefinition(DataDefinition):

    layout_params_func: Callable[...,LayoutParameters] = None # type: ignore
    split_manager: SampleSplitManager = field(default_factory=lambda: DictSampleSplitManager({}))
    get_masks_func: Optional[Callable[[],dict[str,Any]]] = None

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
        if self.layout_params_func is None:
            from datavac.config.layout_params import LayoutParameters
            self.layout_params_func=LayoutParameters
        from datavac.config.sample_splits import get_flow_names
        self.sample_descriptors=FunctionLazyDict(
            getter=lambda name: SampleDescriptor(
                name=name,
                description=f'Sample descriptor for flow {name.split('-- ',maxsplit=1)[1]}.',
                info_columns=self.split_manager.get_split_table_columns(name.split('-- ',maxsplit=1)[1])),
            keylister=lambda: [f'SplitTable -- {f}' for f in get_flow_names()])
        self.subsample_references = FunctionLazyDict(
            getter=lambda name: self._subsample_reference(name),
            keylister=lambda: self._subsample_reference_names())
            
    def get_layout_params_table(self, lp_group) -> pd.DataFrame:
        """Returns the layout parameters for the given layout parameter group."""
        # TODO: Don't need to load all of them here
        from datavac.config.layout_params import LP
        return LP()._tables_by_meas[lp_group]
    def get_layout_params_table_names(self) -> list[str]:
        """Returns a list of all layout parameter group names."""
        # TODO: Don't need to load all of them here
        from datavac.config.layout_params import LP
        return list(LP()._tables_by_meas.keys())
    
    #def get_split_table_columns(self, flow_name: str) -> list[DVColumn]:
    #    """Returns the columns of the split table for the given flow name."""
    #    raise NotImplementedError("This method should be implemented in subclasses to return the split table columns.")
    #def get_flow_names(self) -> list[str]:
    #    """Returns a list of all flow names (each flow name will correspond to a split table)."""
    #    #raise NotImplementedError("This method should be implemented in subclasses to return the flow names.")
    #    return []

    @cache
    def _get_mask_yaml(self):
        import yaml
        from datavac.config.project_config import PCONF
        yaml_path = PCONF().CONFIG_DIR / 'masks.yaml'  # type: ignore
        assert yaml_path.exists(), f"ProjectConfig.get_masks() called but masks.yaml file not found at {yaml_path}"
        with open(yaml_path, 'r') as f: # type: ignore
            mask_yaml = yaml.safe_load(f)
        return mask_yaml
    
    def get_masks(self) -> dict[str, Any]:
        """Each value in output dict should be of the form returned by make_fullwafer_diemap()."""
        if self.get_masks_func: return self.get_masks_func()
        else:
            masks={}
            for mask_name, mask_info in self._get_mask_yaml()['array_maps'].items():
                from datavac.io.make_diemap import make_fullwafer_diemap
                masks[mask_name] = make_fullwafer_diemap(
                    name=mask_name, **mask_info['args'])
            return masks
            
    
    def _subsample_reference(self, name: str) -> SubSampleReference:
        @dataclass(kw_only=True)
        class SSR_MaskRef(SubSampleReference):
            unique_for_mask: str
            def get_constraints(self) -> Sequence[Constraint]:
                from sqlalchemy import UniqueConstraint, ForeignKeyConstraint
                from datavac.database.db_structure import DBSTRUCT
                masksetid = DBSTRUCT().get_sample_reference_dbtable('MaskSet').c['MaskSet']
                return [UniqueConstraint('MaskSet',self.unique_for_mask),
                        ForeignKeyConstraint(['MaskSet'],[masksetid])]
            
            def transform(self, table: DataFrame, sample_info: dict[str,Any], conn: Connection):
                table=super().transform(table, sample_info=sample_info, conn=conn)
                from sqlalchemy import select
                from datavac.database.db_structure import DBSTRUCT
                reftab=DBSTRUCT().get_subsample_reference_dbtable(self.name)
                mapping=dict(conn.execute(select(reftab.c[self.unique_for_mask],reftab.c[self.key_column.name])\
                            .where(reftab.c['MaskSet'] == sample_info['MaskSet'])).all()) # type: ignore
                table[self.key_column.name] = table[self.unique_for_mask].map(mapping)
        if name == 'Dies':
            return SSR_MaskRef(
                name='Dies', unique_for_mask='DieXY',
                description='A subsample reference for dies in a semiconductor sample.',
                key_column=DVColumn('dieid', 'int32', 'The unique identifier for each die.'),
                info_columns=[DVColumn('MaskSet', 'string', 'The mask set used for the die.'),
                              DVColumn('DieXY', 'string', 'The XY label of the die.'),
                              DVColumn('DieX', 'int32', 'The X coordinate of the die.'),
                              DVColumn('DieY', 'int32','The Y coordinate of the die.'),
                              DVColumn('DieCenterA [mm]','float64','Coordinate A (increase right when notch left) of the die center in mm'),
                              DVColumn('DieCenterB [mm]','float64','Coordinate B (increase up when notch left) of the die center in mm'),
                              DVColumn('DieRadius [mm]', 'int32', 'The radius of the die from wafer center in mm'),
                              DVColumn('DieComplete', 'boolean', 'Whether the die is complete or partial')])
        elif 'LayoutParams -- ' in name:
            lpg=name.split(" -- ")[-1]
            lpt=self.get_layout_params_table(lpg)
            #return SSR_MaskRef(
            return SubSampleReference(
                name=name,#unique_for_mask='Structure',
                description=f'Layout parameters for "{lpg}".',
                key_column=DVColumn('Structure', 'string', 'The structure measured'),
                info_columns=[DVColumn(c,lpt[c].dtype.name,description=c) for c in lpt.columns
                              if c not in ['Structure','Site']])
        else:
            raise ValueError(f"Subsample reference '{name}' not found in data definition.")
    def _subsample_reference_names(self) -> list[str]:
        """Returns a list of all subsample reference names."""
        return ['Dies'] + [f'LayoutParams -- {lp_group}' for lp_group in self.get_layout_params_table_names()]
        
    
    @property
    def SAMPLE_COLNAME(self):
        return self.sample_identifier_column.name
    
    @property
    def ALL_SAMPLE_COLNAMES(self):
        return [self.SAMPLE_COLNAME] +\
               [c.name for c in self.sample_info_columns] + \
               [sr.key_column.name for sr in self.sample_references.values()]
    
    def ALL_SAMPLELOAD_COLNAMES(self, trove_name: str):
        return self.ALL_SAMPLE_COLNAMES\
            + [c.name for c in self.troves[trove_name].load_info_columns]
    
    def populate_initial(self, conn: Optional[Connection] = None):
        super().populate_initial(conn=conn)

        from datavac.database.db_semidev import upload_mask_info, update_layout_params, upload_splits, update_split_tables
        upload_mask_info(self.get_masks(),conn=conn)
        update_layout_params(conn=conn, dump_extractions_and_analyses=False)
        update_split_tables(conn=conn)
        #for sp_name in self.split_manager.get_flow_names(force_external=True):
        #    from datavac.database.db_semidev import create_split_table_view
        #    create_split_table_view(sp_name, conn=conn, just_DDL_string=False)
               
def DDEF() -> DataDefinition:
    """Returns the current data definition."""
    from datavac.config.project_config import PCONF
    return PCONF().data_definition