from __future__ import annotations
from functools import cache
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

from datavac.util.dvlogging import logger
from datavac.trove import ReaderCard, Trove
from datavac.trove.trove_util import get_cached_glob
from datavac.util.util import only

if TYPE_CHECKING:
    import pandas as pd
    from datavac.io.measurement_table import MultiUniformMeasurementTable

class NoDataFromFileException(Exception): pass

@dataclass(kw_only=True)
class FolderTroveReaderCard(ReaderCard):
    glob: str
    read_from_filename_regex: str
    read_so_far_args: dict[str, str] = field(default_factory=dict)
    post_reads: Sequence[Callable[['pd.DataFrame'],None]] = ((lambda x: None),)

@dataclass
class SingleFolderTrove(Trove):

    read_dir: Path = None # type: ignore
    """The directory to read from.  If not specified, will default to the environment variable DATAVACUUM_READ_DIR."""

    filecache_context_manager: Optional[Callable[[], Any]] = None
    """A context manager to use for caching file reads.  If not specified, will default to a null context manager."""

    def __post_init__(self):
        if self.read_dir is None:
            if (rd:=os.environ.get('DATAVACUUM_READ_DIR')) is None:
                raise Exception(f"read_dir not specified for {self.__class__.__name__}, "
                                "and no environment variable DATAVACUUM_READ_DIR to default to.")
            else: self.read_dir = Path(rd)

    def read(self,
             only_meas_groups:Optional[list[str]]=None, only_sampleload_info:dict[str,Any]={},
             only_file_names:Optional[list[str]]=None, info_already_known:dict={},
             cached_glob:Optional[Callable[[Path,str],list[Path]]]=None)\
                -> tuple[dict[str,dict[str,'MultiUniformMeasurementTable']],
                                        dict[str,dict[str,str]]]:
        """Read the folder and return the data and material/load information."""
        return self.read_folder_nonrecursive(
            folder=self.read_dir, trove_name=self.name, cached_glob=cached_glob,
            filecache_context_manager=self.filecache_context_manager,
            only_meas_groups=only_meas_groups, only_sampleload_info=only_sampleload_info,
            only_file_names=only_file_names, info_already_known=info_already_known,)

    @staticmethod
    def read_folder_nonrecursive(folder:Path,trove_name:str,
                   only_meas_groups:Optional[list[str]]=None, only_sampleload_info:dict[str,Any]={},
                   only_file_names:Optional[list[str]]=None, info_already_known:dict={},
                   cached_glob:Optional[Callable[[Path,str],list[Path]]]=None,
                   filecache_context_manager:Optional[Callable[[],Any]]=None)\
                 -> tuple[dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]]]:

        import pandas as pd
        from datavac.io.measurement_table import MultiUniformMeasurementTable
        #if only_meas_groups is not None:
        #    ensure_meas_group_sufficiency(only_meas_groups, required_only=False)
        assert folder.exists(), f"Can't find folder {str(folder)}"

        # Collations of data and material/load information
        sample_to_mg_to_data={}
        sample_to_sampleload_info={}

        # For caching the glob lists and regex-extraction from the filename
        cached_glob=cached_glob or get_cached_glob()
        cached_match=cache(re.match)

        # Go through all files in the folder
        for f in cached_glob(folder,'*'):

            # Set up the caching manager for files
            with (filecache_context_manager or nullcontext)():

                # Ignore obviously not-needed files
                if f.name.startswith("~"): continue # Ignore temp files on windows
                if f.name.startswith("IGNORE"): continue # Ignore on request
                if only_file_names and f.name not in only_file_names: continue

                # Go through the measurement groups, and the readers available for each
                # And see if any of the readers can read this file
                found_mgs, not_found_mgs=[],[]
                from datavac.config.project_config import PCONF
                for mg_name, mg in PCONF().data_definition.measurement_groups.items():
                    if only_meas_groups and mg_name not in only_meas_groups: continue
                    for reader_card in cast(list[FolderTroveReaderCard],mg.reader_cards.get(trove_name,[])):
                        pattern=reader_card.glob
                        if f in cached_glob(folder,pattern):

                            # Read the information from the filename
                            regex=reader_card.read_from_filename_regex
                            if (mo:=cached_match(regex,f.name)) is None:
                                raise Exception(f"Couldn't parse {f.name} with regex {regex}")
                            read_from_filename=mo.groupdict() if regex is not None else {}
                            read_info_so_far=dict(**read_from_filename,**info_already_known)
                            completer=PCONF().data_definition.sample_info_completer
                            read_info_so_far=completer(read_info_so_far)

                            # If we can already rule out this file based on its name, skip it
                            if any(read_info_so_far[k] not in only_sampleload_info[k]
                                   for k in only_sampleload_info if k in read_info_so_far):
                                continue

                            # List this meas group among those which will have been checked for this file
                            not_found_mgs.append(mg_name) # will remove if found
                            found=False

                            # Read data
                            try:
                                read_dfs=reader_card.read(file=f,
                                        mg_name=mg_name,only_sampleload_info=only_sampleload_info,
                                        read_info_so_far=read_info_so_far)
                                if not len(read_dfs): raise NoDataFromFileException('Empty sheet')
                            except NoDataFromFileException as e: continue

                            # Some reader functions (operating on single-material files) return simply
                            # a sequence of dataframes.  Others which operate on files with multiple
                            # materials per file return a dictionary of dataframes mapping material
                            # name to a sequence of dataframes. Handle both by converting the former to
                            # the latter, with the assumption we've already garnered the material name
                            # from the folder or filename
                            SAMPLE_COLNAME=PCONF().data_definition.SAMPLE_COLNAME
                            ALL_SAMPLELOAD_COLNAMES=PCONF().data_definition.ALL_SAMPLELOAD_COLNAMES(trove_name)
                            sample_to_read_dfs:dict[str,list[pd.DataFrame]]=\
                                read_dfs if type(read_dfs) is dict else {read_info_so_far[SAMPLE_COLNAME]:read_dfs} # type: ignore

                            # For each material and sequence of dataframs
                            for sample,read_dfs in sample_to_read_dfs.items():
                                # Check if read_dfs it contains any material/load information
                                # If so, check that info for consistency with what we've already gathered
                                # and add altogether to read_info_so_far_with_data
                                # Of course the sample is also material/load info, so include that as well
                                # And for convenience, add the sample to each dataframe so post_reads have access to it
                                read_info_so_far_with_data=read_info_so_far.copy()
                                for k in ALL_SAMPLELOAD_COLNAMES:
                                    if any(k in read_df.columns for read_df in read_dfs):
                                        from_data=only(set([read_df[k].iloc[0] for read_df in read_dfs]))
                                    else: from_data=None
                                    if k==SAMPLE_COLNAME:
                                        if from_data is not None: assert sample==from_data
                                        else:
                                            for read_df in read_dfs:
                                                read_df[SAMPLE_COLNAME]=pd.Series([sample]*len(read_df),dtype='string')
                                        from_data=sample
                                    if from_data is not None:
                                        if k in read_info_so_far_with_data:
                                            assert read_info_so_far[k]==from_data, f"From data {from_data} vs {read_info_so_far[k]}"
                                        else:
                                            read_info_so_far_with_data[k]=from_data
                                read_info_so_far_with_data=completer(read_info_so_far_with_data)

                                # If altogether this info is enough to rule out this data, skip incorporating it
                                if any(read_info_so_far_with_data[k] not in only_sampleload_info[k]
                                       for k in only_sampleload_info if k in read_info_so_far_with_data):
                                    continue

                                # Do the post_reads
                                for read_df in read_dfs:
                                    for post_read in reader_card.post_reads: post_read(read_df)

                                # Form MeasurementTable from the data
                                # And add some useful externally ensured columns
                                read_data=MultiUniformMeasurementTable.from_read_data(read_dfs=read_dfs,meas_group=mg)
                                try: relpath=f.relative_to(os.environ['DATAVACUUM_READ_DIR'])
                                except: relpath=f
                                read_data['FilePath']=pd.Series([
                                    str(relpath.as_posix())]*len(read_data),dtype='string')
                                read_data['FileName']=pd.Series([str(f.name)]*len(read_data),dtype='string')
                                read_data[SAMPLE_COLNAME]=pd.Series([sample]*len(read_data),dtype='string')
                                if 'DieXY' in read_data:
                                    read_data['FQSite']=read_data[SAMPLE_COLNAME]+'/'+read_data['DieXY']+'/'+read_data['Site']

                                # Add any material LUT information to the data
                                for k,v in read_info_so_far.get('material_lut',{}).get(sample,{}).items():
                                    logger.debug(f"Applying {k}={v} to data")
                                    if type(v)==str:
                                            read_data[k]=pd.Series([v]*len(read_data),dtype='string')
                                    else:
                                        raise NotImplementedError(f"What dtype for {k} which seems to be {str(type(v))}?")

                                # Collate this MeasurementTable onto any others from the same material and meas group
                                sample_to_mg_to_data[sample]=sample_to_mg_to_data.get(sample,{})
                                if (existing_data:=sample_to_mg_to_data[sample].get(mg_name,None)):
                                    sample_to_mg_to_data[sample][mg_name]=existing_data+read_data
                                else:
                                    sample_to_mg_to_data[sample][mg_name]=read_data

                                # Also collate any material/load information
                                sample_to_sampleload_info[sample]=sample_to_sampleload_info.get(sample,{})
                                for k in ALL_SAMPLELOAD_COLNAMES:
                                    if not (v:=read_info_so_far_with_data.get(k,None)):
                                        raise Exception(f"Missing info to know '{k}' for {str(f)}")
                                    if (existing_value:=sample_to_sampleload_info[sample].get(k,None)):
                                        assert existing_value==v, f"Different values for {k} seen: {existing_value} vs {v}"
                                    sample_to_sampleload_info[sample][k]=v

                                # If we've made it this far, some data was found for this meas group in this file
                                found=True

                            # Track what's been found or not for this file
                            if found: found_mgs.append(not_found_mgs.pop(-1))

                # If this file got checked for any meas groups, notify about what's been found
                if len(found_mgs+not_found_mgs):
                    logger.info(f"In {f.relative_to(folder)}, found {found_mgs}")
        return sample_to_mg_to_data, sample_to_sampleload_info