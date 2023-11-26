import os
import re
import pandas as pd
from datetime import datetime
from functools import wraps, lru_cache
from pathlib import Path

import yaml

from datavac.io.measurement_table import MultiUniformMeasurementTable
from datavac.logging import logger
from datavac.util import import_modfunc, check_dtypes

with open(Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"meta_reader.yaml",'r') as f:
    meta_reader_yaml=yaml.safe_load(f)
with open(Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"db_schema.yaml",'r') as f:
    db_schema_yaml=yaml.safe_load(f)
ALL_MATERIAL_COLUMNS=[db_schema_yaml['materials']['full_name'],*db_schema_yaml['materials']['info_columns']]
READ_DIR=Path(os.environ['DATAVACUUM_READ_DIR'])

class MissingFolderInfoException(Exception): pass
class NoDataFromFileException(Exception): pass

def read_folder_nonrecursive(folder: str,
               only_meas_groups=None,only_material_info={},
               only_file_names:list[str]=None) -> dict:

    if not (folder:=Path(folder)).is_absolute():
        folder=READ_DIR/folder
    assert folder.exists(), f"Can't find folder {str(folder)}"

    # Read any auxiliary information in the folder
    read_from_folder={}
    for rname,rinfo in meta_reader_yaml.get('read_from_folder',{}).items():
        assert (rcount:=rinfo.get('count','required')) in ['required','optional']
        if len(potential_finds:=list(folder.glob(rinfo['filter'])))==1:
            match rinfo['read_action']:
                case 'store_path':
                    read_from_folder[rname]=potential_finds[0]
                case 'store_basename':
                    read_from_folder[rname]=potential_finds[0].name.split(".")[0]
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
    with import_modfunc(meta_reader_yaml['caching_manager'])():
        for f in folder.iterdir():
            if f.name.startswith("~"): continue # Ignore temp files on windows
            if only_file_names and f.name not in only_file_names: continue
            found_mgs, not_found_mgs=[],[]
            for meas_group, mg_info in meta_reader_yaml['measurement_groups'].items():
                if only_meas_groups and meas_group not in only_meas_groups: continue
                for reader in mg_info['readers']:
                    for pattern, reader_func_dotpath in reader['reader_func'].items():
                        #print(pattern, reader_func_dotpath, list(cached_glob(pattern))
                        if f in cached_glob(pattern):
                            #logger.info(f"Looking for measurement group {meas_group} in {f.relative_to(folder)}")
                            regex=reader.get('read_from_filename_regex',
                                             meta_reader_yaml.get('read_from_filename_regex',None))
                            if (mo:=cached_match(regex,f.name)) is None:
                                raise Exception(f"Couldn't parse {f.name} with regex {regex}")
                            read_from_filename=mo.groupdict() if regex is not None else {}
                            read_info_so_far=dict(**read_from_filename,**read_from_folder)
                            meas_type=import_modfunc(mg_info['meas_type'])(**mg_info.get('meas_type_args',{}))

                            #print(f"Restriction {only_material_info}, currently known {read_info_so_far}")
                            if any(read_info_so_far[k] not in only_material_info[k]
                                   for k in only_material_info if k in read_info_so_far):
                                continue
                            not_found_mgs.append(meas_group) # will remove if found

                            try:
                                read_dfs=import_modfunc(reader_func_dotpath)(file=f,
                                        meas_type=meas_type,meas_group=meas_group,
                                        **reader['fixed_args'],
                                        **{k:read_info_so_far[v] for k,v in reader['read_so_far_args'].items()})
                            except NoDataFromFileException as e:
                                #logger.info(f"In {f.relative_to(folder)}, found nothing for '{meas_group}'")
                                continue
                            read_data=MultiUniformMeasurementTable.from_read_data(read_dfs=read_dfs,
                                meas_group=meas_group,meas_type=meas_type)
                            read_data['FileName']=pd.Series([
                                str(f.relative_to(os.environ['DATAVACUUM_READ_DIR']).as_posix())]*len(read_data),dtype='string')

                            if not (matname:=read_info_so_far.get(db_schema_yaml['materials']['full_name'],None)):
                                # Gotta read this from the data then and check uniqueness
                                raise NotImplementedError

                            matname_to_mg_to_data[matname]=matname_to_mg_to_data.get(matname,{})
                            if (existing_data:=matname_to_mg_to_data[matname].get(meas_group,None)):
                                matname_to_mg_to_data[matname][meas_group]=existing_data+read_data
                            else:
                                matname_to_mg_to_data[matname][meas_group]=read_data

                            matname_to_material_info[matname]=matname_to_material_info.get(matname,
                                           {db_schema_yaml['materials']['full_name']:matname})
                            for k in db_schema_yaml['materials']['info_columns']:
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

def perform_analysis(matname_to_mg_to_data):
    for matname, mg_to_data in matname_to_mg_to_data.items():
        for mg, data in mg_to_data.items():
            logger.debug(f"{mg} analysis")
            for pre_analysis in meta_reader_yaml['measurement_groups'][mg].get('pre_analysis',[]):
                if type(pre_analysis) is list:
                    pre_analysis,*args=pre_analysis
                else:
                    pre_analysis,*args=pre_analysis,
                import_modfunc(pre_analysis)(data,*args)
            data.analyze()
            check_dtypes(data.scalar_table)

#@wraps(read_folder_nonrecursive,updated=('__name__',))
def read_and_analyze_folders(folders, *args, **kwargs) -> dict:

    for folder in folders:
        if not (folder:=Path(folder)).is_absolute():
            folder=READ_DIR/folder
        assert folder.exists(), f"Can't find folder {str(folder)}"

        contributions=[]

        dirs=[folder]
        while len(dirs):
            curdir=dirs.pop(0)
            for file in curdir.iterdir():
                if file.is_dir():
                    dirs.append(file)
            try:
                logger.info(f"Reading in {curdir.relative_to(READ_DIR)}")
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

    perform_analysis(matname_to_mg_to_data)
    return matname_to_mg_to_data,matname_to_material_info