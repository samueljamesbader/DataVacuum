import os
import re
import pandas as pd
from functools import lru_cache
from pathlib import Path

from datavac.io.measurement_table import MultiUniformMeasurementTable
from datavac.util.logging import logger
from datavac.util.util import import_modfunc
from datavac.util.tables import check_dtypes
from datavac.util.conf import CONFIG

FULLNAME_COL=CONFIG['database']['materials']['full_name']
ALL_MATERIAL_COLUMNS=[CONFIG['database']['materials']['full_name'],
                      *CONFIG['database']['materials']['info_columns']]
if 'DATAVACUUM_READ_DIR' in os.environ:
    READ_DIR=Path(os.environ['DATAVACUUM_READ_DIR'])

class MissingFolderInfoException(Exception): pass
class NoDataFromFileException(Exception): pass

def read_folder_nonrecursive(folder: str,
               only_meas_groups=None,only_material_info={},
               only_file_names:list[str]=None) -> dict:

    if only_meas_groups is not None:
        ensure_meas_group_sufficiency(only_meas_groups, required_only=False)

    if not (folder:=Path(folder)).is_absolute():
        folder=READ_DIR/folder
    assert folder.exists(), f"Can't find folder {str(folder)}"

    # Read any auxiliary information in the folder
    read_from_folder={}
    for rname,rinfo in CONFIG['meta_reader'].get('read_from_folder',{}).items():
        assert (rcount:=rinfo.get('count','required')) in ['required','optional']
        if len(potential_finds:=list(folder.glob(rinfo['filter'])))==1:
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

    matname_to_mg_to_data={}
    matname_to_material_info={}
    cached_glob=lru_cache(lambda patt: list(folder.glob(patt)))
    cached_match=lru_cache(re.match)
    with import_modfunc(CONFIG['meta_reader']['caching_manager'])():
        for f in folder.iterdir():
            if f.name.startswith("~"): continue # Ignore temp files on windows
            if f.name.startswith("IGNORE"): continue # Ignore on request
            if only_file_names and f.name not in only_file_names: continue
            found_mgs, not_found_mgs=[],[]
            for meas_group, mg_info in CONFIG['measurement_groups'].items():
                if only_meas_groups and meas_group not in only_meas_groups: continue
                for reader in mg_info['readers']:
                    if True:
                    #for pattern, reader_pointer in reader[''].items():
                        pattern=reader['template']['glob'];reader_func_dotpath=reader['template']['function']
                        #print(pattern, reader_func_dotpath, list(cached_glob(pattern))
                        if f in cached_glob(pattern):
                            #logger.info(f"Looking for measurement group {meas_group} in {f.relative_to(folder)}")
                            regex=reader.get('read_from_filename_regex',
                                             CONFIG['meta_reader'].get('read_from_filename_regex',None))
                            if (mo:=cached_match(regex,f.name)) is None:
                                raise Exception(f"Couldn't parse {f.name} with regex {regex}")
                            read_from_filename=mo.groupdict() if regex is not None else {}
                            read_info_so_far=dict(**read_from_filename,**read_from_folder)
                            meas_type=CONFIG.get_meas_type(meas_group)

                            #print(f"Restriction {only_material_info}, currently known {read_info_so_far}")
                            if any(read_info_so_far[k] not in only_material_info[k]
                                   for k in only_material_info if k in read_info_so_far):
                                continue
                            not_found_mgs.append(meas_group) # will remove if found

                            try:
                                read_dfs=import_modfunc(reader_func_dotpath)(file=f,
                                        meas_type=meas_type,meas_group=meas_group,
                                        **reader['template'].get('default_args',{}),
                                        **reader.get('fixed_args',{}),
                                        **{k:read_info_so_far[v] for k,v in reader['template']['read_so_far_args'].items()})
                                if not len(read_dfs): raise NoDataFromFileException('Empty sheet')
                                if 'post_read' in reader:
                                    for read_df in read_dfs: import_modfunc(reader['post_read'])(read_df)
                            except NoDataFromFileException as e:
                                #logger.info(f"In {f.relative_to(folder)}, found nothing for '{meas_group}'")
                                continue
                            read_data=MultiUniformMeasurementTable.from_read_data(read_dfs=read_dfs,
                                meas_group=meas_group,meas_type=meas_type)
                            try: relpath=f.relative_to(os.environ['DATAVACUUM_READ_DIR'])
                            except: relpath=f
                            read_data['FilePath']=pd.Series([
                                str(relpath.as_posix())]*len(read_data),dtype='string')
                            read_data['FileName']=pd.Series([str(f.name)]*len(read_data),dtype='string')

                            if not (matname:=read_info_so_far.get(FULLNAME_COL,None)):
                                # Gotta read this from the data then and check uniqueness
                                raise NotImplementedError
                            for k,v in read_info_so_far.get('material_lut',{}).get(matname,{}).items():
                                logger.debug(f"Applying {k}={v} to data")
                                if type(v)==str:
                                        read_data[k]=pd.Series([v]*len(read_data),dtype='string')
                                else:
                                    raise NotImplementedError(f"What dtype for {k} which seems to be {str(type(v))}?")

                            matname_to_mg_to_data[matname]=matname_to_mg_to_data.get(matname,{})
                            if (existing_data:=matname_to_mg_to_data[matname].get(meas_group,None)):
                                matname_to_mg_to_data[matname][meas_group]=existing_data+read_data
                            else:
                                matname_to_mg_to_data[matname][meas_group]=read_data

                            matname_to_material_info[matname]=matname_to_material_info.get(matname,
                                           {FULLNAME_COL:matname})
                            for k in CONFIG['database']['materials']['info_columns']:
                                if not (v:=read_info_so_far.get(k,None)):
                                    # Gotta read this from the data then and check uniqueness
                                    raise NotImplementedError
                                if (existing_value:=matname_to_material_info[matname].get(k,None)):
                                    assert existing_value==v, f"Different values for {k} seen: {existing_value} vs {v}"
                                matname_to_material_info[matname][k]=v
                            found_mgs.append(not_found_mgs.pop(-1))
            if len(found_mgs+not_found_mgs):
                logger.info(f"In {f.relative_to(folder)}, found {found_mgs}, didn't find {not_found_mgs}.")
    return matname_to_mg_to_data, matname_to_material_info

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
def read_and_analyze_folders(folders, *args, **kwargs) -> dict:

    folders=[READ_DIR/folder if not Path(folder).is_absolute() else Path(folder) for folder in folders]
    for folder in folders:
        assert folder.exists(), f"Can't find folder {str(folder)}"

    contributions=[]
    for folder in folders:
        dirs=[folder]
        while len(dirs):
            curdir=dirs.pop(0)
            for file in curdir.iterdir():
                if file.is_dir():
                    if 'IGNORE' not in str(file):
                        dirs.append(file)
            try:
                try:
                    logger.info(f"Reading in {curdir.relative_to(READ_DIR)}")
                except ValueError:
                    logger.info(f"Reading in {curdir}")
                contributions.append(read_folder_nonrecursive(curdir, *args, **kwargs))
            except MissingFolderInfoException as e:
                logger.info(f"Skipping {curdir.relative_to(READ_DIR)} because {str(e)}")

    matname_to_mg_to_data={}
    matname_to_material_info={}
    for matname_to_mg_to_dirdata, matname_to_dirinfo in contributions:
        for matname, mg_to_dirdata in matname_to_mg_to_dirdata.items():

            matname_to_mg_to_data[matname]=matname_to_mg_to_data.get(matname,{})
            for mg, dirdata in mg_to_dirdata.items():
                if (existing_data:=matname_to_mg_to_data[matname].get(mg,None)):
                    matname_to_mg_to_data[matname][mg]=existing_data+dirdata
                else:
                    matname_to_mg_to_data[matname][mg]=dirdata

            if (existing_info:=matname_to_material_info.get(matname,None)):
                assert existing_info==matname_to_dirinfo[matname],\
                    f"Different material infos {existing_info} vs {matname_to_dirinfo[matname]}"
            matname_to_material_info[matname]=matname_to_dirinfo[matname]

    for mg_to_data in matname_to_mg_to_data.values():
        for data in mg_to_data.values():
            data.defrag()

    perform_extraction(matname_to_mg_to_data)
    return matname_to_mg_to_data,matname_to_material_info