from __future__ import annotations
from functools import cache
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping, Optional, Sequence, cast
from collections import UserDict
from typing import Dict, Any

from datavac.config.data_definition import DVColumn
from datavac.util.dvlogging import logger, time_it
from datavac.trove import ReaderCard, Trove, TroveIncrementalTracker
from datavac.trove.trove_util import PathWithMTime, get_cached_glob
from datavac.util.util import only
from datavac.util.util import returner_context

if TYPE_CHECKING:
    import pandas as pd
    from datavac.io.measurement_table import MultiUniformMeasurementTable
    from sqlalchemy import MetaData, Table, Connection
    import pandas as pd

class NoDataFromFileException(Exception): pass

class SingleFolderTroveIncrementalTracker(UserDict[str, dict[str, Any]], TroveIncrementalTracker):
    """Incremental tracker for SingleFolderTrove; just a UserDict[str, dict[str, Any]]."""
    pass


@dataclass(kw_only=True)
class FolderTroveReaderCard(ReaderCard):
    glob: str
    read_from_filename_regex: str = ''
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

    def iter_read(self,
             only_meas_groups:Optional[list[str]]=None, only_sampleload_info:dict[str,Any]={},
             only_file_names:Optional[list[str]]=None, info_already_known:dict={},
             cached_glob:Optional[Callable[[Path,str],list[PathWithMTime]]]=None,
             incremental:bool=False,
             exception_callback:Optional[Callable[[str,Exception],None]]=None)\
                -> Generator[tuple[str,dict[str,dict[str,'MultiUniformMeasurementTable']],
                                        dict[str,dict[str,str]],
                                        dict[str,dict[str,Any]]]]:
        """Read the folder and return the data and material/load information."""
        ## If incremental, get the modified times of all files in the folder from the database
        if incremental: raise NotImplementedError("Incremental read not yet implemented for SingleFolderTrove")
        #if incremental:
        #    from datavac.database.db_util import get_engine_ro
        #    from datavac.database.db_structure import DBSTRUCT
        #    from sqlalchemy import select
        #    file_tab=DBSTRUCT().get_trove_dbtables(trove_name)['files']
        #    with get_engine_ro().connect() as conn:
        #        db_modtimes_result=conn.execute(select(file_tab.c['FilePath'],file_tab.c['ModifiedTime']).where(
        #            file_tab.c.FilePath.in_([str(f.as_posix()) for f in cached_glob(folder,'*')]))).fetchall()
        #        db_modtimes:dict[str,int]=dict(db_modtimes_result)# type: ignore
        try:
            yield str(self.read_dir),*self.read_folder_nonrecursive(
                folder=self.read_dir, trove_name=self.name, cached_glob=cached_glob,
                filecache_context_manager=self.filecache_context_manager,incremental_tracker=None,
                only_meas_groups=only_meas_groups, only_sampleload_info=only_sampleload_info,
                only_file_names=only_file_names, info_already_known=info_already_known,),{}
        except Exception as e:
            if exception_callback: exception_callback(str(self.read_dir),e)
            else: raise e

    @staticmethod
    def read_folder_nonrecursive(folder:Path,trove_name:str,
                   only_meas_groups:Optional[list[str]]=None, only_sampleload_info:dict[str,Any]={},
                   only_file_names:Optional[list[str]]=None, info_already_known:dict={},
                   cached_glob:Optional[Callable[[Path,str],list[PathWithMTime]]]=None,
                   incremental_tracker:Optional[SingleFolderTroveIncrementalTracker]=None,
                   super_folder_for_filepaths: Optional[Path] = None, 
                   filecache_context_manager:Optional[Callable[[],Any]]=None)\
                 -> tuple[dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]]]:

        import pandas as pd
        from datavac.io.measurement_table import MultiUniformMeasurementTable
        #if only_meas_groups is not None:
        #    ensure_meas_group_sufficiency(only_meas_groups, required_only=False)
        assert folder.exists(), f"Can't find folder {str(folder)}"
        if super_folder_for_filepaths is None: super_folder_for_filepaths=folder
        assert super_folder_for_filepaths is not None

        # Collations of data and material/load information
        sample_to_mg_to_data={}
        sample_to_sampleload_info={}

        # For caching the glob lists and regex-extraction from the filename
        cached_glob=cached_glob or get_cached_glob(super_folder_for_filepaths)
        cached_match=cache(re.match)

        # Go through all files in the folder
        for f in cached_glob(folder,'*'):
            if f.cached_is_dir: continue

            if incremental_tracker is not None:
                fstr=str(f.as_posix())
                if fstr in incremental_tracker:
                    if incremental_tracker[fstr]['DBModifiedTime']==f.mtime_ns:
                        logger.info(f"Skipping {f} because it hasn't been modified since last read")
                        incremental_tracker[fstr]['State']='Unchanged'
                        continue
                    else:
                        logger.info(f"File {f} has been modified since last read (DB: {incremental_tracker[fstr]['DBModifiedTime']} vs local: {f.mtime_ns}), including in read")
                        incremental_tracker[fstr]['State']='Modified'
                        incremental_tracker[fstr]['LocalModifiedTime']=f.mtime_ns
                else:
                    logger.info(f"File {f} is new since last read, including in read")
                    incremental_tracker[fstr]={'State':'New', 'LocalModifiedTime':f.mtime_ns, 'DBMeasGroups':[]}

            #if incremental and (f.mtime_ns is not None) and (f.mtime_ns==db_modtimes.get(str(f.as_posix()),None)):
            #    logger.info(f"Skipping {f} because it hasn't been modified since last read")
            #    continue

            # Set up the caching manager for files
            with (filecache_context_manager or nullcontext)():

                # Ignore obviously not-needed files
                if f.cached_is_dir: continue
                if f.name.startswith("~"): continue # Ignore temp files on windows
                if only_file_names and f.name not in only_file_names: continue
                if (not only_file_names) and f.name.startswith("IGNORE"): continue # Ignore on request

                # Go through the measurement groups, and the readers available for each
                # And see if any of the readers can read this file
                found_mgs, not_found_mgs=[],[]
                from datavac.config.project_config import PCONF
                for mg_name, mg in PCONF().data_definition.measurement_groups.items():
                    if only_meas_groups and mg_name not in only_meas_groups: continue
                    for reader_card in cast(list[FolderTroveReaderCard],mg.reader_cards.get(trove_name,[])):
                        if hasattr(reader_card,'allowed_folders'):
                            if not any(fn in reader_card.allowed_folders for fn in folder.parts): continue # type: ignore
                        pattern=reader_card.glob
                        if f in cached_glob(folder,pattern):

                            # Read the information from the filename
                            regex=reader_card.read_from_filename_regex
                            if (mo:=cached_match(regex,f.name)) is None:
                                raise Exception(f"Couldn't parse {f.name} with regex {regex}")
                            read_from_filename=mo.groupdict() if regex is not None else {}
                            for k,v in read_from_filename.items():
                                assert (k not in info_already_known) or (info_already_known[k]==v),\
                                    f"Info {k}={v} from filename contradicts "\
                                    f"info already known {k}={info_already_known[k]} "\
                                    f"from folder for file {str(f)}"
                            read_info_so_far=read_from_filename|info_already_known|{'ModifiedTime': f.mtime_ns}
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
                                read_dfs=reader_card.read(file=super_folder_for_filepaths/f,
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
                                read_dfs if type(read_dfs) is dict\
                                    else {read_info_so_far[SAMPLE_COLNAME]:
                                          (read_dfs if not isinstance(read_dfs,pd.DataFrame) else [read_dfs])} # type: ignore

                            # For each material and sequence of dataframs
                            for sample,read_dfs in sample_to_read_dfs.items():
                                # Check if read_dfs it contains any material/load information
                                # If so, check that info for consistency with what we've already gathered
                                # and add altogether to read_info_so_far_with_data
                                # Of course the sample is also material/load info, so include that as well
                                read_info_so_far_with_data=read_info_so_far.copy()
                                for k in ALL_SAMPLELOAD_COLNAMES:
                                    if any(k in read_df.columns for read_df in read_dfs):
                                        from_data=only(set([read_df[k].iloc[0] for read_df in read_dfs]))
                                    else: from_data=None
                                    if k==SAMPLE_COLNAME:
                                        if from_data is not None: assert sample==from_data
                                        from_data=sample
                                    if from_data is not None:
                                        if k in read_info_so_far_with_data:
                                            assert read_info_so_far[k]==from_data, f"From data {from_data} vs {read_info_so_far[k]}"
                                        else:
                                            read_info_so_far_with_data[k]=from_data
                                read_info_so_far_with_data=completer(read_info_so_far_with_data)

                                # For convenience, add the sampleload back to each dataframe so post_reads have access to it
                                for k in read_info_so_far_with_data:
                                    for read_df in read_dfs:
                                        if k not in read_df.columns:
                                            read_df[k]=pd.Series([read_info_so_far_with_data[k]]*len(read_df)).convert_dtypes()

                                # If altogether this info is enough to rule out this data, skip incorporating it
                                if any(read_info_so_far_with_data[k] not in only_sampleload_info[k]
                                       for k in only_sampleload_info if k in read_info_so_far_with_data):
                                    continue

                                # Do the post_reads
                                with time_it("Post-reads",.01):
                                    for read_df in read_dfs:
                                        for post_read in reader_card.post_reads: post_read(read_df)

                                # Form MeasurementTable from the data
                                # And add some useful externally ensured columns
                                read_data=MultiUniformMeasurementTable.from_read_data(read_dfs=read_dfs,meas_group=mg)
                                read_data['FilePath']=pd.Series([
                                    str(f.as_posix())]*len(read_data),dtype='string')
                                read_data['FileName']=pd.Series([str(f.name)]*len(read_data),dtype='string')
                                read_data[SAMPLE_COLNAME]=pd.Series([sample]*len(read_data),dtype='string')
                                if ('DieXY' in read_data) and ('Site' in read_data):
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
                    logger.info(f"In {(super_folder_for_filepaths/f).relative_to(folder)}, found {found_mgs}")
        if incremental_tracker is not None:
            folder_str=str((folder.relative_to(super_folder_for_filepaths)).as_posix())
            for fstr,info in incremental_tracker.items():
                if (info['State'] is None) and fstr.startswith(folder_str+'/') and '/' not in fstr[len(folder_str)+1:]:
                    logger.info(f"File {fstr} is no longer present, marking as removed")
                    info['State']='Removed'
            db_sample_names=set([s for info in incremental_tracker.values() for s in info.get('DBSampleNames',[])])
            from datavac.config.data_definition import DDEF
            for sample in db_sample_names:
                if sample not in sample_to_sampleload_info:
                    sample_to_mg_to_data[sample]={}
                    sample_to_sampleload_info[sample]=None
        return sample_to_mg_to_data, sample_to_sampleload_info

    def additional_tables(self, int_schema: str, metadata: MetaData, load_tab: Table) -> tuple[dict[str, Table], dict[str, Table]]:
        from sqlalchemy import Table, Column, INTEGER, VARCHAR, ForeignKey, UniqueConstraint, BigInteger
        from datavac.database.db_structure import _CASC
        #addtabs=super().additional_tables(int_schema, metadata, load_tab=load_tab) # causes trouble for using this for classic folder trove which inherits from this, so skipping for now
        addtabs={}
        t= metadata.tables.get(int_schema+f'.TTTT --- Files_{self.name}')
        addtabs['files']=file_tab= t if (t is not None) else \
            Table(f'TTTT --- Files_{self.name}', metadata,
                  Column('fileid',INTEGER,primary_key=True,autoincrement=True),
                  Column('FilePath',VARCHAR,nullable=False,unique=True),
                  Column('FileName',VARCHAR,nullable=False,unique=False),
                  # store as nanoseconds since epoch for easy comparison with filesystem timestamps.  Must be bigint
                  Column('ModifiedTime',BigInteger,nullable=False),
                  # Not really used by SingleFolderTrove but useful for ClassicFolderTrove which reuses this function
                  Column('ReadgroupName',VARCHAR,nullable=False),
                  schema=int_schema)
        t= metadata.tables.get(int_schema+f'.TTTT --- FileLoads_{self.name}')
        addtabs['fileloads']=fileload_tab= t if (t is not None) else \
            Table(f'TTTT --- FileLoads_{self.name}', metadata,
                  Column('fileid',INTEGER,ForeignKey(file_tab.c.fileid,**_CASC),nullable=False),
                  Column('loadid',INTEGER,ForeignKey(load_tab.c.loadid,**_CASC),nullable=False),
                  UniqueConstraint('fileid','loadid',name=f'uq_file_load_{self.name}'),
                  schema=int_schema)
        return {'files':addtabs['files']}, {'fileloads':addtabs['fileloads']}
    
    def trove_reference_columns(self) -> Mapping[DVColumn,str]:
        return {DVColumn('fileid','int','Source file identifier'): 'files'}
    
    def transform(self, df: pd.DataFrame, loadid: int, sample_info:dict[str,Any],
                  conn:Optional[Connection]=None, readgrp_name:str|None=None):
        # df will have columns 'FilePath' and 'ModificationTime'.
        # First, update the "TTTT -- Files" with all the new FilePaths (in one query), retrieving the fileids for the new entries.
        # Then, replace the FilePath column in df with the corresponding fileids.
        # Then update the "TTTT -- FileLoads" to connect the loadid to the fileids for the files that were just added.
        if 'FilePath' not in df.columns:
            raise Exception("ClassicFolderTrove.transform() expects a column 'FilePath' in the dataframe.")
        file_tab=self.dbtables('files')
        from datavac.database.db_connect import get_engine_rw
        from sqlalchemy.dialects.postgresql import insert as pgsql_insert
        with (returner_context(conn) if conn is not None else get_engine_rw().begin()) as conn:
            assert conn is not None
            fp_to_mt=dict(df[['FilePath','ModifiedTime']].drop_duplicates().set_index('FilePath', verify_integrity=True)['ModifiedTime'])
            fp_to_mtfn={fp:{'ModifiedTime': mt, 'FileName': os.path.basename(fp)} for fp,mt in fp_to_mt.items()}
            # moved cleaning to elsewher
            #uniq_fps_clean={fp:str(Path(fp).relative_to(self.read_dir)) for fp in fp_to_mt.keys()}
            uniq_fps_clean={fp:fp for fp in fp_to_mt.keys()}
            update_info=[{'FilePath': cfp,'ModifiedTime':int(fp_to_mtfn[rfp]['ModifiedTime']),
                          'FileName':fp_to_mtfn[rfp]['FileName'],'ReadgroupName':readgrp_name}
                      for rfp,cfp in uniq_fps_clean.items()]
            fileid_from_clean_lookup={v:k for k,v in conn.execute(pgsql_insert(file_tab)\
                         .values(update_info)\
                         .on_conflict_do_update(index_elements=['FilePath'],set_=file_tab.c)\
                         .returning(file_tab.c.fileid, file_tab.c.FilePath)).fetchall()}
            oldfps=[fp for fp in uniq_fps_clean.values() if fp not in fileid_from_clean_lookup]
            if len(oldfps):
                fileid_from_clean_lookup.update({v:k for k,v in conn.execute(file_tab.select()\
                    .where(file_tab.c.FilePath.in_(oldfps))).fetchall()})
            fileid_from_raw_lookup={uniq_fps_clean[fp]:fileid_from_clean_lookup[uniq_fps_clean[fp]] for fp in fp_to_mt}
            df['fileid']=df['FilePath'].map(fileid_from_raw_lookup)
            df.drop(columns=['FilePath'], inplace=True)

            fileload_tab=self.dbtables('fileloads')
            conn.execute(pgsql_insert(fileload_tab)\
                            .values([{'fileid': fid, 'loadid': loadid} for fid in fileid_from_raw_lookup.values()]))
    def affected_meas_groups_and_filters(self, samplename: Any, comp: SingleFolderTroveIncrementalTracker,
                                         data_by_mg: dict[str,MultiUniformMeasurementTable], conn: Optional[Connection] = None)\
            -> tuple[list[str], list[str], dict[str,Sequence[Any]]]:
        affected_files_maybe_not_in_dbm=[fstr for fstr,info in comp.items()
                        if info['State'] in ('Modified','Removed') # if 'New', then will have to be in data_by_mg to matter
                        and samplename in info['DBSampleNames']]
        unaffected_files=[fstr for fstr,info in comp.items()
                          if info['State']=='Unchanged'
                          and samplename in info['DBSampleNames']]
        affected_mgs=set([mg_name for mg_name in data_by_mg]+\
                         [mg_name for af in affected_files_maybe_not_in_dbm for mg_name in comp[af]['DBMeasGroups']])
        mgs_with_old_data=set([mg_name for info in comp.values()
                               if info['State']!='New' and samplename in info['DBSampleNames']
                                   for mg_name in info['DBMeasGroups']])
        
        #from datavac.database.db_connect import get_engine_ro
        #from datavac.database.db_structure import DBSTRUCT
        #from sqlalchemy import select
        #from datavac.config.project_config import PCONF
        #samplename_col=PCONF().data_definition.SAMPLE_COLNAME
        #with (returner_context(conn) if conn is not None else get_engine_ro().begin()) as conn:
        #    sampletab=DBSTRUCT().get_sample_dbtable()
        #    loadtab=DBSTRUCT().get_trove_dbtables(self.name)['loads']
        #    fileloadtab=DBSTRUCT().get_trove_dbtables(self.name)['fileloads']
        #    filetab=DBSTRUCT().get_trove_dbtables(self.name)['files']
        #    query=select(loadtab.c.MeasGroup).distinct().select_from(loadtab\
        #        .join(sampletab, loadtab.c.sampleid==sampletab.c.sampleid)\
        #        .join(fileloadtab, fileloadtab.c.loadid==loadtab.c.loadid) \
        #        .join(filetab, filetab.c.fileid==fileloadtab.c.fileid)) \
        #        .where(sampletab.c[samplename_col]==samplename)\
        #        .where(filetab.c.FilePath.in_(affected_files))
        #    affected_mgs_with_old_data=[row[0] for row in conn.execute(query).fetchall()]

        affected_mgs_with_old_data=[mg_name for mg_name in affected_mgs if mg_name in mgs_with_old_data]
        affected_mgs_without_old_data=[mg_name for mg_name in data_by_mg if mg_name not in mgs_with_old_data]
        return affected_mgs_with_old_data,affected_mgs_without_old_data, {'FilePath': unaffected_files}
        

fqsite_col = DVColumn('FQSite','string','Fully-qualified site name')