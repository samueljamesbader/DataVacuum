from __future__ import annotations
from dataclasses import dataclass, field
import os
import re
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping, Optional, Sequence, Union, cast
from functools import lru_cache, cache
from pathlib import Path

from datavac.config.data_definition import DVColumn
from datavac.trove import ReaderCard, Trove, TroveIncrementalTracker
from datavac.trove.folder_aux_info import FolderAuxInfoReader, MissingFolderInfoException
from datavac.trove.single_folder_trove import FolderTroveReaderCard, SingleFolderTrove, SingleFolderTroveIncrementalTracker
from datavac.trove.trove_util import PathWithMTime, get_cached_glob
from datavac.util.dvlogging import logger

if TYPE_CHECKING:
    from datavac.io.measurement_table import MultiUniformMeasurementTable
    from sqlalchemy import MetaData, Table, Connection
    import pandas as pd


class ClassicFolderTroveIncrementalTracker(TroveIncrementalTracker):
    """Incremental tracker for ClassicFolderTrove; wraps a SingleFolderTroveIncrementalTracker."""

    def __init__(self):
        self._inner = SingleFolderTroveIncrementalTracker()

    @property
    def inner(self) -> SingleFolderTroveIncrementalTracker:
        return self._inner


@dataclass
class ClassicFolderTroveReaderCard(FolderTroveReaderCard): pass

@dataclass
class ClassicFolderTrove(Trove):

    read_dir: Path = None # type: ignore
    """The root directory to read from.  If not specified, will default to the environment variable DATAVACUUM_READ_DIR."""

    ignore_folder_regex: str = r'.*IGNORE.*'
    """A regex to match folder names which should be ignored.  If not specified, will default to '.*IGNORE.*'."""

    folder_aux_info_reader: FolderAuxInfoReader = field(default_factory=FolderAuxInfoReader)
    """A reader for auxiliary information in folders.  If not specified, will default to an empty reader."""

    filecache_context_manager: Optional[Callable[[], Any]] = None
    """A context manager to use for caching file reads.  If not specified, will default to a null context manager."""

    natural_grouping: Optional[str] = None
    """If specified, the top-level folder can be narrowed down by this name of a sample/load information column.
    
    e.g. if the read_dir is organized by each top-level folder containing one 'Lot', and 'Lot' is a column in
    the DataDefinition sample information or the Trove load information, then setting this to 'Lot' will enable
    iter_read() to visit only the relevant top-level folder instead of ALL of them.
    """

    prompt_for_readall: bool = True
    """If True, will prompt the user to confirm reading if no specific top-level folder is clear from arguments."""

    cli_expander: dict[str,dict[str,Any]] = field(default_factory=lambda:\
              {'only_folders': dict(name_or_flags=['--folder'],type=str, nargs='+',
                                help='Restrict to these top-level folders.  If not specified, will read all top-level folders.'),
              'only_file_names': dict(name_or_flags=['--file'],type=str, nargs='+',
                                  help='Restrict to these file names'),
              'dont_prompt_readall': dict(name_or_flags=['--dont_prompt_all','-dpa'], action='store_true',
                                help='If set, will *not* prompt the user to confirm reading all folders if no specific top-level folder is specified.'),
               })

    def __post_init__(self):
        if self.read_dir is None:
            if 'DATAVACUUM_READ_DIR' not in os.environ:
                raise Exception("read_dir not specified for ClassicFolderTrove, "\
                                "and no environment variable DATAVACUUM_READ_DIR to default to.")
            else: self.read_dir = Path(os.environ['DATAVACUUM_READ_DIR'])
    
    def iter_read(self, # super Trove.read() signature
                   only_meas_groups:Optional[list[str]]=None,
                   only_sampleload_info:dict[str,Any]={},
                   info_already_known:dict[str,Any]={},
                   incremental:bool=False,
                   # FolderTrove-specific arguments
                   only_file_names:Optional[list[str]]=None, only_folders: Optional[Sequence[Path]]=None,
                   cached_glob:Optional[Callable[[Path,str],list[PathWithMTime]]]=None, dont_recurse:bool=False,
                   dont_prompt_readall:bool=False, exception_callback: Optional[Callable[[str,Exception],None]]=None)\
             -> Generator[tuple[str,dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]],ClassicFolderTroveIncrementalTracker|None]]:
        """Reads data from the trove read_dir.

        Args:
            only_meas_groups, only_sampleload_info, info_already_known: Same as Trove.iter_read()
            only_file_names: If set, only reads files with these names.
            only_folders: If set, only reads the specified folders.
                If not set, self.natural_grouping can be used to restrict the read
            cached_glob: a function which takes a folder and a pattern, and returns a list of paths matching the pattern.
            dont_recurse: If True, will not recurse into subfolders.
            dont_prompt_readall: If True, will (override self.prompt_for_readall) and not prompt the user
                to confirm reading all folders if no specific top-level folder is specified.
        """

        # Attempt to narrow down the read_dir to specific folder(s)
        if only_folders is None:
            if self.natural_grouping and (self.natural_grouping in only_sampleload_info):
                folders=only_sampleload_info[self.natural_grouping]
            else:
                if (not dont_prompt_readall) and self.prompt_for_readall:
                    if not (input(f'No folder or top-level restriction,'\
                                  ' continue to read EVERYTHING? [y/n] ')\
                                .strip().lower()=='y'): yield "",{},{},None
                folders=self.get_natural_grouping_factors()
        else: folders=only_folders

        # Check that folders are valid paths before we get to far
        folders=[self.read_dir/folder if not Path(folder).is_absolute() else Path(folder) for folder in folders]
        for folder in folders: assert folder.exists(), f"Can't find folder {str(folder)}"

        folders_by_toplevel={}
        for folder in folders:
            tl=folder.relative_to(self.read_dir).parts[0]
            folders_by_toplevel[tl]=folders_by_toplevel.get(tl,[]) + [folder]
        for tl, folders in folders_by_toplevel.items():

            if incremental:
                comparison_to_prior_loads = ClassicFolderTroveIncrementalTracker()
                # Get file modification times from the database to compare against filesystem modification times for incremental reading
                from datavac.database.db_util import get_engine_ro
                from datavac.database.db_structure import DBSTRUCT
                from datavac.config.data_definition import DDEF
                from sqlalchemy import select
                file_tab=DBSTRUCT().get_trove_dbtables(self.name)['files']
                fileload_tab=DBSTRUCT().get_trove_dbtables(self.name)['fileloads']
                load_tab=DBSTRUCT().get_trove_dbtables(self.name)['loads']
                sample_tab=DBSTRUCT().get_sample_dbtable()
                SAMPLENAME_COL=DDEF().SAMPLE_COLNAME
                with get_engine_ro().connect() as conn:
                    # Note: by joining to fileload_tab, this should get only files that are associated to an existing load!
                    # That's important because nothing currently exists to delete file_tab entries when loads are deleted,
                    # and we don't want to compare against files that were previously loaded but have since been deleted from the database.
                    db_modtimes_result=conn.execute(select(file_tab.c['FilePath'],file_tab.c['ModifiedTime'],fileload_tab.c['loadid'],
                                                           load_tab.c['MeasGroup'],sample_tab.c[SAMPLENAME_COL])\
                                                           .select_from(file_tab.join(fileload_tab).join(load_tab).join(sample_tab))\
                                                           .where(file_tab.c['ReadgroupName']==tl)).fetchall()
                    for row in db_modtimes_result:
                        if row[0] not in comparison_to_prior_loads.inner:
                            comparison_to_prior_loads.inner[row[0]] = {'DBModifiedTime':row[1],'State':None,'DBMeasGroups':[row[3]],'DBSampleNames':[row[4]],}
                        else:
                            comparison_to_prior_loads.inner[row[0]]['DBMeasGroups'].append(row[3])
                            comparison_to_prior_loads.inner[row[0]]['DBSampleNames'].append(row[4])
            else: comparison_to_prior_loads = None


            # Store info known about the data at each folder in the tree up to the top-level
            @cache
            def get_info_from_folder(insp_folder:Path) -> dict[str,Any]:
                if insp_folder==self.read_dir:
                    if (not self.natural_grouping) or (self.natural_grouping in info_already_known):
                        return info_already_known
                    else:
                        return dict(**info_already_known,**{self.natural_grouping:tl})
                info_from_parent=get_info_from_folder(insp_folder.parent)
                logger.debug(f"Looking around in {insp_folder.relative_to(self.read_dir)}")
                info_from_this_folder=self.folder_aux_info_reader.read(
                    insp_folder,cached_glob=cached_glob,super_folder=self.read_dir,
                    info_already_known=info_from_parent)
                logger.debug("Info known down to this level of the tree: "+str(info_from_this_folder))
                return info_from_this_folder

            # Recurse down the folder structure reading in each
            try:
                cached_glob=cached_glob or get_cached_glob(self.read_dir)
                contributions=[]
                for folder in folders:
                    dirs=[folder]
                    while len(dirs):
                        curdir=dirs.pop(0)
                        if not dont_recurse:
                            for file in cached_glob(curdir,'*'):
                                if file.cached_is_dir:
                                    if re.match(self.ignore_folder_regex,file.name):
                                        logger.debug(f"Ignoring {file} because matches \"{self.ignore_folder_regex}\".")
                                    else: dirs.append(self.read_dir/file)
                        try:
                            try:
                                logger.info(f"Reading in {curdir.relative_to(self.read_dir)}")
                            except ValueError:
                                logger.info(f"Reading in {curdir}")
                            contributions.append(SingleFolderTrove.read_folder_nonrecursive(
                                folder=curdir, trove_name=self.name,
                                incremental_tracker=comparison_to_prior_loads.inner if incremental else None, # type: ignore
                                only_meas_groups=only_meas_groups, only_sampleload_info=only_sampleload_info,
                                only_file_names=only_file_names, info_already_known=get_info_from_folder(curdir),
                                super_folder_for_filepaths=self.read_dir,
                                cached_glob=cached_glob, filecache_context_manager=self.filecache_context_manager))
                        except MissingFolderInfoException as e:
                            logger.info(f"Skipping {curdir.relative_to(self.read_dir)} because {str(e)}")

                # Combine the contributions from all folders
                matname_to_mg_to_data={}
                matname_to_matload_info={}
                for matname_to_mg_to_dirdata, matname_to_dirinfo in contributions:
                    for matname, mg_to_dirdata in matname_to_mg_to_dirdata.items():

                        matname_to_mg_to_data[matname]=matname_to_mg_to_data.get(matname,{})
                        for mg, dirdata in mg_to_dirdata.items():
                            if (existing_data:=matname_to_mg_to_data[matname].get(mg,None)):
                                matname_to_mg_to_data[matname][mg]=existing_data+dirdata
                            else:
                                matname_to_mg_to_data[matname][mg]=dirdata

                        if (existing_info:=matname_to_matload_info.get(matname,None)):
                            assert existing_info==matname_to_dirinfo[matname],\
                                f"Different material infos {existing_info} vs {matname_to_dirinfo[matname]}"
                        else:
                            matname_to_matload_info[matname]=matname_to_dirinfo[matname]

                # Defragment the dataframe in each measurement group for performance
                for mg_to_data in matname_to_mg_to_data.values():
                    for data in mg_to_data.values():
                        data.defrag()

                if incremental:
                    assert comparison_to_prior_loads is not None # for type checker
                    for fstr, info in comparison_to_prior_loads.inner.items():
                        if info['State'] is None: info['State']='Removed'
                    logger.debug(f"Incremental read summary for {tl}:")
                    for fstr in comparison_to_prior_loads.inner:
                        logger.debug(f"  {comparison_to_prior_loads.inner[fstr]['State'].ljust(12)} {fstr}")
                yield str(tl), matname_to_mg_to_data,matname_to_matload_info, comparison_to_prior_loads
            except Exception as e:
                if exception_callback: exception_callback(str(tl), e)
                else: raise e

    def get_natural_grouping_factors(self) -> list[Any]:
        return sorted([f.name for f in self.read_dir.glob('*')
                        if 'IGNORE' not in f.name], reverse=True)
    
    def trove_reference_columns(self) -> Mapping[DVColumn,str]:
        return SingleFolderTrove.trove_reference_columns(cast(SingleFolderTrove,self))

    def additional_tables(self, int_schema: str, metadata: MetaData, load_tab: Table) -> tuple[dict[str, Table], dict[str, Table]]:
        sft_data,sft_syst=SingleFolderTrove.additional_tables(cast(SingleFolderTrove,self),int_schema, metadata, load_tab)

        from sqlalchemy import Table, Column, INTEGER, VARCHAR, ForeignKey, UniqueConstraint, BigInteger
        from datavac.database.db_structure import _CASC
        syst=sft_syst.copy()
        t= metadata.tables.get(int_schema+f'.TTTT --- Folders_{self.name}')
        syst['folders']=fold_tab= t if (t is not None) else \
            Table(f'TTTT --- Folders_{self.name}', metadata,
                  Column('FolderPath',VARCHAR,nullable=False,unique=True),
                  Column('FolderAux',VARCHAR,nullable=False),
                  Column('ReadgroupName',VARCHAR,nullable=False),
                  schema=int_schema)
        return sft_data, syst
        
    def transform(self, df: pd.DataFrame, loadid: int, sample_info: dict[str, Any], conn: Connection | None = None, readgrp_name: str | None = None):
        super().transform(df, loadid, sample_info, conn)
        return SingleFolderTrove.transform(cast(SingleFolderTrove,self),
                                           df, loadid, sample_info, conn, readgrp_name)
    
    def affected_meas_groups_and_filters(self, samplename: Any, comp: ClassicFolderTroveIncrementalTracker,
                                         data_by_mg: dict[str,MultiUniformMeasurementTable], conn: Optional[Connection] = None)\
            -> tuple[list[str], list[str], dict[str,Sequence[Any]]]:
        return SingleFolderTrove.affected_meas_groups_and_filters(cast(SingleFolderTrove,self), samplename, comp.inner, data_by_mg, conn)