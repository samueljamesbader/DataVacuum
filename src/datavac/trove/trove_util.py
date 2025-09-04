from __future__ import annotations
from pathlib import Path
from functools import cache
from typing import Union, TYPE_CHECKING
import argparse
from fnmatch import fnmatch


if TYPE_CHECKING:
    from datavac.io.measurement_table import MultiUniformMeasurementTable

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


def quick_read_filename(filename:Union[Path,str],extract=True,**kwargs)\
                -> tuple[dict[str,dict[str,MultiUniformMeasurementTable]],dict[str,dict[str,str]]]:
    from datavac.trove.classic_folder_trove import ClassicFolderTrove
    from datavac.config.project_config import PCONF
    trove = list(PCONF().data_definition.troves.values())[0]
    assert isinstance(trove, ClassicFolderTrove),\
        f"Quick read only works for ClassicFolderTrove, got {type(trove)}"
    if not (filename:=Path(filename)).is_absolute():
        filename=trove.read_dir/filename
    assert filename.exists(), f"Can't find file {str(filename)}"
    if filename.is_dir():
        mt2mg2dat,mt2ml=trove.read(only_folders=[filename], only_file_names=None,
                               info_already_known=kwargs, dont_recurse=True)
    else:
        folder=filename.parent
        mt2mg2dat,mt2ml=trove.read(only_folders=[folder], only_file_names=[filename.name],
                               info_already_known=kwargs, dont_recurse=True)
    if extract:
        from datavac.measurements.meas_util import perform_extraction
        perform_extraction(mt2mg2dat)
    return mt2mg2dat,mt2ml

def cli_quick_read_filename(*args):
    parser=argparse.ArgumentParser(description="Quickly read a single file and extract its data")
    parser.add_argument('filename',type=str,help="The filename to read")
    parser.add_argument('--no-extract',action='store_true',help="Don't extract the data")
    namespace=parser.parse_args(args)
    mt2mg2dat,mt2ml=quick_read_filename(namespace.filename,extract=(not namespace.no_extract))


def ensure_meas_group_sufficiency(meas_groups, required_only=False, on_error='raise', just_extraction=False):
    meas_groups_prev=meas_groups
    while True:
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
        meas_groups=list(set(meas_groups).union(set(add_meas_groups1).union(add_meas_groups2)))
        if sorted(meas_groups_prev)==sorted(meas_groups): return meas_groups
        else: meas_groups_prev=meas_groups