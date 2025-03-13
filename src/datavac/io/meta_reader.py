import os
import re
from fnmatch import fnmatch
from typing import Union

import pandas as pd
from functools import lru_cache, cache
from pathlib import Path

from datavac.io.measurement_table import MultiUniformMeasurementTable
from datavac.util.logging import logger
from datavac.util.util import import_modfunc, only
from datavac.util.tables import check_dtypes
from datavac.util.conf import CONFIG

FULLNAME_COL=CONFIG['database']['materials']['full_name']
ALL_MATERIAL_COLUMNS=[CONFIG['database']['materials']['full_name'],
                      *CONFIG['database']['materials'].get('info_columns',{})]
ALL_LOAD_COLUMNS=CONFIG['database'].get('loads',{}).get('info_columns',[])
ALL_MATLOAD_COLUMNS=[*ALL_MATERIAL_COLUMNS,*ALL_LOAD_COLUMNS]

if os.environ.get('DATAVACUUM_READ_DIR'):
    READ_DIR=Path(os.environ['DATAVACUUM_READ_DIR'])

class MissingFolderInfoException(Exception): pass
class NoDataFromFileException(Exception): pass

def get_cached_glob():
    @cache
    def _cached_iterdir(folder:Path):
        #logger.debug(f"Running iterdir for {folder}")
        return list(folder.iterdir())
    @cache
    def cached_glob(folder:Path,patt:str):
        paths=_cached_iterdir(folder)
        return [p for p in paths if fnmatch(p,patt)]
    return cached_glob

def quick_read_filename(filename:Union[Path,str],extract=True,**kwargs) -> tuple[dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]]]:
    if not (filename:=Path(filename)).is_absolute():
        filename=READ_DIR/filename
    assert filename.exists(), f"Can't find file {str(filename)}"
    folder=filename.parent
    mt2mg2dat,mt2ml=read_folder_nonrecursive(folder,only_file_names=[filename.name],already_read_from_folder=kwargs)
    if extract: perform_extraction(mt2mg2dat)
    return mt2mg2dat,mt2ml

def read_folder_nonrecursive(folder: Union[Path,str],
               only_meas_groups=None,only_matload_info={},
               only_file_names:list[str]=None,already_read_from_folder=None,
               cached_glob=None) -> dict:

    if only_meas_groups is not None:
        ensure_meas_group_sufficiency(only_meas_groups, required_only=False)

    if not (folder:=Path(folder)).is_absolute():
        folder=READ_DIR/folder
    assert folder.exists(), f"Can't find folder {str(folder)}"

    if cached_glob is None:
        cached_glob=get_cached_glob()

    # Read any auxiliary information in the folder
    #logger.debug(f"Reading aux info in {folder}")
    read_from_folder=already_read_from_folder.copy() if already_read_from_folder else {}
    for rname,rinfo in CONFIG.get('meta_reader',{}).get('read_from_folder',{}).items():
        assert (rcount:=rinfo.get('count','required')) in ['required','optional']
        if len(potential_finds:=list(cached_glob(folder,(rinfo['filter']))))==1:
            match rinfo['read_action']:
                case 'store_path':
                    read_from_folder[rname]=potential_finds[0]
                case 'store_basename':
                    read_from_folder[rname]=potential_finds[0].name.split(".")[0]
                case 'apply_material_lut':
                    read_from_folder['material_lut']=\
                        pd.read_csv(potential_finds[0]).set_index(FULLNAME_COL).to_dict(orient='index')
                case _:
                    raise Exception(f"What action is {rinfo['read_action']}")
        else:
            if rcount=='required':
                raise MissingFolderInfoException(
                    f"'Found {len(potential_finds)} options for {rname} "\
                        f"(filter '{rinfo['filter']}') in {str(folder.relative_to(READ_DIR))}'")

    # Collations of data and material/load information
    matname_to_mg_to_data={}
    matname_to_matload_info={}

    # For caching the regex-extraction from the filename
    cached_match=lru_cache(re.match)

    # Set up the caching manager for files
    with import_modfunc(CONFIG.get('meta_reader',{}).get('caching_manager','contextlib:nullcontext'))():

        # Go through all files in the folder
        for f in cached_glob(folder,'*'):

            # Ignore obviously not-needed files
            if f.name.startswith("~"): continue # Ignore temp files on windows
            if f.name.startswith("IGNORE"): continue # Ignore on request
            if only_file_names and f.name not in only_file_names: continue

            # Go through the measurement groups, and the readers available for each
            # And see if any of the readers can read this file
            found_mgs, not_found_mgs=[],[]
            for meas_group, mg_info in CONFIG['measurement_groups'].items():
                if only_meas_groups and meas_group not in only_meas_groups: continue
                for reader in mg_info['readers']:
                    pattern=reader.get('glob',reader['template'].get('glob')); assert pattern is not None
                    reader_func_dotpath=reader['template']['function']
                    if f in cached_glob(folder,pattern):

                        # Read the information from the filename
                        regex=reader.get('read_from_filename_regex',
                                         CONFIG.get('meta_reader',{}).get('read_from_filename_regex','.*'))
                        if (mo:=cached_match(regex,f.name)) is None:
                            raise Exception(f"Couldn't parse {f.name} with regex {regex}")
                        read_from_filename=mo.groupdict() if regex is not None else {}
                        read_info_so_far=dict(**read_from_filename,**read_from_folder)
                        completer=reader.get('matload_info_completer',
                                         CONFIG.get('meta_reader',{}).get('matload_info_completer',None))
                        completer=import_modfunc(completer) if completer else lambda x:x
                        read_info_so_far=completer(read_info_so_far)
                        meas_type=CONFIG.get_meas_type(meas_group)

                        # If we can already rule out this file based on its name, skip it
                        if any(read_info_so_far[k] not in only_matload_info[k]
                               for k in only_matload_info if k in read_info_so_far):
                            continue

                        # List this meas group among those which will have been checked for this file
                        not_found_mgs.append(meas_group) # will remove if found
                        found=False

                        # Read data
                        try:
                            read_dfs=import_modfunc(reader_func_dotpath)(file=f,
                                    meas_type=meas_type,meas_group=meas_group,only_matload_info=only_matload_info,
                                    **dict(reader['template'].get('default_args',{}),
                                           **reader.get('fixed_args',{})),
                                    **{k:read_info_so_far[v] for k,v in reader['template'].get('read_so_far_args',{}).items()})
                            if not len(read_dfs): raise NoDataFromFileException('Empty sheet')
                        except NoDataFromFileException as e: continue

                        # Some reader functions (operating on single-material files) return simply
                        # a sequence of dataframes.  Others which operate on files with multiple
                        # materials per file return a dictionary of dataframes mapping material
                        # name to a sequence of dataframes. Handle both by converting the former to
                        # the latter, with the assumption we've already garnered the material name
                        # from the folder or filename
                        matname_to_read_dfs=read_dfs if type(read_dfs) is dict \
                            else {read_info_so_far[FULLNAME_COL]:read_dfs}

                        # For each material and sequence of dataframs
                        for matname,read_dfs in matname_to_read_dfs.items():
                            # Check if read_dfs it contains any material/load information
                            # If so, check that info for consistency with what we've already gathered
                            # and add altogether to read_info_so_far_with_data
                            # Of course the matname is also material/load info, so include that as well
                            # And for convenience, add the matname to each dataframe so post_read has access to it
                            read_info_so_far_with_data=read_info_so_far.copy()
                            for k in ALL_MATLOAD_COLUMNS:
                                if any(k in read_df.columns for read_df in read_dfs):
                                    from_data=only(set([read_df[k].iloc[0] for read_df in read_dfs]))
                                else: from_data=None
                                if k==FULLNAME_COL:
                                    if from_data is not None: assert matname==from_data
                                    else:
                                        for read_df in read_dfs:
                                            read_df[FULLNAME_COL]=pd.Series([matname]*len(read_df),dtype='string')
                                    from_data=matname
                                if from_data is not None:
                                    if k in read_info_so_far_with_data:
                                        assert read_info_so_far[k]==from_data, f"From data {from_data} vs {read_info_so_far[k]}"
                                    else:
                                        read_info_so_far_with_data[k]=from_data
                            read_info_so_far_with_data=completer(read_info_so_far_with_data)

                            # If altogether this info is enough to rule out this data, skip incorporating it
                            if any(read_info_so_far_with_data[k] not in only_matload_info[k]
                                   for k in only_matload_info if k in read_info_so_far_with_data):
                                continue

                            # Do the post_read
                            if 'post_read' in reader:
                                for read_df in read_dfs: import_modfunc(reader['post_read'])(read_df)

                            # Form MeasurementTable from the data
                            # And add some useful externally ensured columns
                            read_data=MultiUniformMeasurementTable.from_read_data(read_dfs=read_dfs,
                                meas_group=meas_group,meas_type=meas_type)
                            try: relpath=f.relative_to(os.environ['DATAVACUUM_READ_DIR'])
                            except: relpath=f
                            read_data['FilePath']=pd.Series([
                                str(relpath.as_posix())]*len(read_data),dtype='string')
                            read_data['FileName']=pd.Series([str(f.name)]*len(read_data),dtype='string')
                            read_data[FULLNAME_COL]=pd.Series([matname]*len(read_data),dtype='string')
                            if 'DieXY' in read_data:
                                read_data['FQSite']=read_data[FULLNAME_COL]+'/'+read_data['DieXY']+'/'+read_data['Site']

                            # Add any material LUT information to the data
                            for k,v in read_info_so_far.get('material_lut',{}).get(matname,{}).items():
                                logger.debug(f"Applying {k}={v} to data")
                                if type(v)==str:
                                        read_data[k]=pd.Series([v]*len(read_data),dtype='string')
                                else:
                                    raise NotImplementedError(f"What dtype for {k} which seems to be {str(type(v))}?")

                            # Collate this MeasurementTable onto any others from the same material and meas group
                            matname_to_mg_to_data[matname]=matname_to_mg_to_data.get(matname,{})
                            if (existing_data:=matname_to_mg_to_data[matname].get(meas_group,None)):
                                matname_to_mg_to_data[matname][meas_group]=existing_data+read_data
                            else:
                                matname_to_mg_to_data[matname][meas_group]=read_data

                            # Also collate any material/load information
                            matname_to_matload_info[matname]=matname_to_matload_info.get(matname,{})
                            for k in ALL_MATLOAD_COLUMNS:
                                if not (v:=read_info_so_far_with_data.get(k,None)):
                                    raise Exception(f"Missing info to know '{k}' for {str(f)}")
                                if (existing_value:=matname_to_matload_info[matname].get(k,None)):
                                    assert existing_value==v, f"Different values for {k} seen: {existing_value} vs {v}"
                                matname_to_matload_info[matname][k]=v

                            # If we've made it this far, some data was found for this meas group in this file
                            found=True

                        # Track what's been found or not for this file
                        if found: found_mgs.append(not_found_mgs.pop(-1))

            # If this file got checked for any meas groups, notify about what's been found
            if len(found_mgs+not_found_mgs):
                logger.info(f"In {f.relative_to(folder)}, found {found_mgs}")
    return matname_to_mg_to_data, matname_to_matload_info

def ensure_meas_group_sufficiency(meas_groups, required_only=False, on_error='raise', just_extraction=False):
    add_meas_groups1=CONFIG.get_dependency_meas_groups_for_meas_groups(meas_groups, required_only=required_only)
    missing=[mg for mg in add_meas_groups1 if mg not in meas_groups]
    if on_error=='raise':
        assert len(missing)==0, f"Measurement groups {meas_groups} also require {missing}"+\
                                (" to be checked." if not required_only else ".")
    if just_extraction:
        return list(set(meas_groups).union(set(add_meas_groups1)))

    ans=CONFIG.get_dependent_analyses(meas_groups)
    add_meas_groups2=CONFIG.get_dependency_meas_groups_for_analyses(analyses=ans)
    missing=[mg for mg in add_meas_groups2 if mg not in meas_groups]
    if on_error=='raise':
        assert len(missing)==0, f"Measurements groups {meas_groups} affect analyses {ans}, "\
                                f"which require {missing}"+\
                                (" to be checked." if not required_only else ".")
    return list(set(meas_groups).union(set(add_meas_groups1).union(add_meas_groups2)))

def perform_extraction(matname_to_mg_to_data):
    for matname, mg_to_data in matname_to_mg_to_data.items():
        to_be_extracted=list(mg_to_data.keys())
        ensure_meas_group_sufficiency(to_be_extracted,required_only=True, just_extraction=True)
        while len(to_be_extracted):
            for mg in to_be_extracted:
                deps=CONFIG.get_dependency_meas_groups_for_meas_groups([mg],required_only=False)
                if any(d in to_be_extracted for d in deps): continue

                data=mg_to_data[mg]
                logger.debug(f"{mg} extraction ({matname})")
                for pre_analysis in CONFIG['measurement_groups'][mg].get('pre_analysis',[]):
                    import_modfunc(pre_analysis)(data)
                dep_kws={deps[d]:mg_to_data[d] for d in deps if d in mg_to_data}
                data.analyze(**dep_kws)
                for post_analysis in CONFIG['measurement_groups'][mg].get('post_analysis',[]):
                    import_modfunc(post_analysis)(data)
                check_dtypes(data.scalar_table)

                to_be_extracted.remove(mg)

#@wraps(read_folder_nonrecursive,updated=('__name__',))
def read_and_analyze_folders(folders, *args, cached_glob=None, **kwargs) -> dict:

    folders=[READ_DIR/folder if not Path(folder).is_absolute() else Path(folder) for folder in folders]
    for folder in folders:
        assert folder.exists(), f"Can't find folder {str(folder)}"
    ignore_folders=os.environ.get("DATAVACUUM_IGNORE_FOLDERS",".*IGNORE.*")

    if cached_glob is None:
        cached_glob=get_cached_glob()
    contributions=[]
    for folder in folders:
        dirs=[folder]
        while len(dirs):
            curdir=dirs.pop(0)
            for file in cached_glob(curdir,'*'):
                if file.is_dir():
                    if re.match(ignore_folders,file.name):
                        logger.debug(f"Ignoring {file} because matches \"{ignore_folders}\"."\
                            " Control via environment variable DATAVACUUM_IGNORE_FOLDERS.")
                    else:
                        dirs.append(file)
            try:
                try:
                    logger.info(f"Reading in {curdir.relative_to(READ_DIR)}")
                except ValueError:
                    logger.info(f"Reading in {curdir}")
                contributions.append(read_folder_nonrecursive(curdir, *args, cached_glob=cached_glob, **kwargs))
            except MissingFolderInfoException as e:
                logger.info(f"Skipping {curdir.relative_to(READ_DIR)} because {str(e)}")

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

    for mg_to_data in matname_to_mg_to_data.values():
        for data in mg_to_data.values():
            data.defrag()

    perform_extraction(matname_to_mg_to_data)
    return matname_to_mg_to_data,matname_to_matload_info