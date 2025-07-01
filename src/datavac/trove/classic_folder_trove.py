from __future__ import annotations
from dataclasses import dataclass, field
import os
import re
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union
from functools import lru_cache, cache
from pathlib import Path

from datavac.trove import ReaderCard, Trove
from datavac.trove.folder_aux_info import FolderAuxInfoReader, MissingFolderInfoException
from datavac.trove.single_folder_trove import FolderTroveReaderCard, SingleFolderTrove
from datavac.trove.trove_util import get_cached_glob
from datavac.util.logging import logger

if TYPE_CHECKING:
    from datavac.io.measurement_table import MultiUniformMeasurementTable


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
    read() to visit only the relevant top-level folder instead of ALL of them.
    """

    prompt_for_readall: bool = True
    """If True, will prompt the user to confirm reading if no specific top-level folder is clear from arguments."""

    def __post_init__(self):
        if self.read_dir is None:
            if 'DATAVACUUM_READ_DIR' not in os.environ:
                raise Exception("read_dir not specified for ClassicFolderTrove, "\
                                "and no environment variable DATAVACUUM_READ_DIR to default to.")
            else: self.read_dir = Path(os.environ['DATAVACUUM_READ_DIR'])
    
    def read(self, # super Trove.read() signature
                   only_meas_groups:Optional[list[str]]=None,
                   only_sampleload_info:dict[str,Any]={},
                   info_already_known:dict={},
                   # FolderTrove-specific arguments
                   only_file_names:Optional[list[str]]=None, only_folders: Optional[Sequence[Path]]=None,
                   cached_glob:Optional[Callable[[Path,str],list[Path]]]=None, dont_recurse:bool=False)\
             -> tuple[dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]]]:
        """Reads data from the trove read_dir.

        Args:
            only_meas_groups, only_sampleload_info, info_already_known: Same as Trove.read()
            only_file_names: If set, only reads files with these names.
            only_folders: If set, only reads the specified folders.
                If not set, self.natural_grouping can be used to restrict the read
            cached_glob: a function which takes a folder and a pattern, and returns a list of paths matching the pattern.
            dont_recurse: If True, will not recurse into subfolders.
        """

        # Attempt to narrow down the read_dir to specific folder(s)
        if only_folders is None:
            if self.natural_grouping and (self.natural_grouping in only_sampleload_info):
                folders=only_sampleload_info[self.natural_grouping]
            else:
                if self.prompt_for_readall:
                    if not (input(f'No folder or top-level restriction,'\
                                  ' continue to read EVERYTHING? [y/n] ')\
                                .strip().lower()=='y'): return {},{}
                folders=sorted([f.name for f in self.read_dir.glob('*')
                                if 'IGNORE' not in f.name], reverse=True)
        else: folders=only_folders

        # Check that folders are valid paths before we get to far
        folders=[self.read_dir/folder if not Path(folder).is_absolute() else Path(folder) for folder in folders]
        for folder in folders: assert folder.exists(), f"Can't find folder {str(folder)}"

        # Recurse down the folder structure reading in each
        cached_glob=cached_glob or get_cached_glob()
        contributions=[]
        for folder in folders:
            dirs=[folder]
            while len(dirs):
                curdir=dirs.pop(0)
                if not dont_recurse:
                    for file in cached_glob(curdir,'*'):
                        if file.is_dir():
                            if re.match(self.ignore_folder_regex,file.name):
                                logger.debug(f"Ignoring {file} because matches \"{self.ignore_folder_regex}\".")
                            else: dirs.append(file)
                try:
                    try:
                        logger.info(f"Reading in {curdir.relative_to(self.read_dir)}")
                    except ValueError:
                        logger.info(f"Reading in {curdir}")
                    info_already_known=self.folder_aux_info_reader.read(
                        curdir,cached_glob=cached_glob,super_folder=self.read_dir,info_already_known=info_already_known)
                    contributions.append(SingleFolderTrove.read_folder_nonrecursive(
                        folder=curdir, trove_name=self.name,
                        only_meas_groups=only_meas_groups, only_sampleload_info=only_sampleload_info,
                        only_file_names=only_file_names, info_already_known=info_already_known,
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
                matname_to_matload_info[matname]=matname_to_dirinfo[matname]

        # Defragment the dataframe in each measurement group for performance
        for mg_to_data in matname_to_mg_to_data.values():
            for data in mg_to_data.values():
                data.defrag()

        #perform_extraction(matname_to_mg_to_data)
        logger.critical("Old flow would have done extraction here, is that implemented elsewhere yet?") # TODO: implement extraction flow
        return matname_to_mg_to_data,matname_to_matload_info