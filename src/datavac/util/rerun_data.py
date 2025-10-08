""" Utilities for downstream packages to do regression testing by rerunning specific data.

Warning: these functions are meant to be called from CLI, not from other code.
They will do an unload_my_imports() to reset the get_specific_db_connection_info() cache so they
can ensure use of a regression-test database and not wipe the project database.
So any code following the run of these functions may end up with inconsistent state unless you
(1) do another unload_my_imports()
(2) are not holding any references to datavac objects that were imported before this function was called
The best approach is to call these functions by CLI or in a separate process, e.g. via subprocess.run().
"""
from contextlib import contextmanager
import datetime
import pickle
from typing import Union, Optional

import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path

# WARNING:
# ensure_in_rerun_mode() does an unload_my_imports, so do any datavac imports LOCALLY


timefmt=r"%Y-%m-%d_%H-%M-%S"

in_rerun_mode=False
@contextmanager
def ensure_in_rerun_mode():
    global in_rerun_mode
    if in_rerun_mode: yield

    # Adding in this restriction temporarily to (1) avoid accidental use and (2) ensure connection settings (SSL args) are correct
    from datavac.config.contexts import get_current_context_name
    assert get_current_context_name() == "local", "Rerun data can only be done in local context"

    from datavac.util.dvlogging import logger
    logger.debug("Entering rerun mode, unloading imports and setting up local regtest database.")
    
    # Override the environment variables so get_specific_db_connection_info returns a local demo db named regtest
    dbname='regtest'
    passwd=os.environ.get('DATAVACUUM_TEST_DB_PASS','insecure_default_password')
    prev_env_val = os.environ.get('DATAVACUUM_DB_CONNECTION_STRING')
    try:
        in_rerun_mode=True
        os.environ['DATAVACUUM_DB_CONNECTION_STRING']=\
            f"Server = localhost; Port = 5432; Database = {dbname};" \
            f" Uid = postgres; Password = {passwd}"
        from datavac import unload_my_imports; unload_my_imports(silent=True)

        # Yield
        yield

    # Restore the environment variable
    finally:
        in_rerun_mode=False
        if prev_env_val is not None:
            os.environ['DATAVACUUM_DB_CONNECTION_STRING'] = prev_env_val
        else:
            os.environ.pop('DATAVACUUM_DB_CONNECTION_STRING', None)

def rerun_data():
    """
    Rerun the data in the rerun_data.yaml file
    """
    with ensure_in_rerun_mode():
        from datavac.config.project_config import PCONF
        from datavac.util.dvlogging import logger
        from datavac.database.db_create import ensure_clear_database, create_all
        from datavac.database.db_get import get_data
        from datavac.config.data_definition import DDEF
        from datavac.util.util import only

        PCONF().vault.clear_vault_cache()
        ensure_clear_database()
        create_all()
        #assert only(DDEF().troves.keys()) =='', "Multi-trove not implemented yet for rerun_data"
        yaml_dict=do_the_rerun_uploads(confirm=False)

        PCONF().RERUN_DIR.mkdir(exist_ok=True,parents=False)
        destfile=PCONF().RERUN_DIR/(datetime.datetime.now().strftime(timefmt)+".pkl")
        dat={mgoa:get_data(mgoa,ensure_consistent_order=True)
             for mgoa in list(DDEF().measurement_groups)+list(DDEF().higher_analyses)}
        metadat={'rerun_yaml':yaml_dict,
                 'rerun_time':datetime.datetime.now()}
        with open(destfile, 'wb') as f:
            pickle.dump({'data':dat,'metadata':metadat},f)
        logger.debug(f"Data rerun and saved to {destfile}")
        return dat
    
def do_the_rerun_uploads(confirm=True):
        from datavac.database.db_upload_meas import read_and_enter_data
        from datavac.config.project_config import PCONF

        if confirm:
            from datavac.config.contexts import get_current_context_name
            cc=get_current_context_name()
            ans=input(f"WARNING: You are about to upload data to the '{cc}' database context, not run regression testing.  Are you sure? (y/n) ")
            if ans.lower() != 'y':
                print("Aborting.")
                return
        
        yaml_path = PCONF().CONFIG_DIR/"rerun_data.yaml" # type: ignore
        assert yaml_path.exists(), f"rerun_data.yaml file not found at {yaml_path}"
        with open(yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        for ud in yaml_dict['regression_data']['uploads']:
            read_and_enter_data(**ud)
        return yaml_dict


def _open_rerun_data_file(filename: Optional[Union[str,Path]] = None):
    from datavac.util.dvlogging import logger
    from datavac.config.project_config import PCONF
    RERUN_DIR=PCONF().RERUN_DIR
    
    if filename is None:
        potential_files=list(RERUN_DIR.glob('2*.pkl'))
        if len(potential_files) == 0:
            raise ValueError(f"No rerun data files found in {RERUN_DIR}")
        filename = max(potential_files, key=lambda file: datetime.datetime.strptime(file.stem, timefmt)).stem
        logger.debug(f"Using most recent rerun data file: {filename}.pkl")
    with open(RERUN_DIR/f"{filename.split(".pkl")[0] if type(filename) is str else filename}.pkl", 'rb') as f:
        return pickle.load(f)


def compare_data(newer: Optional[Union[str,dict[str,pd.DataFrame]]]=None,
                 older: Union[str,dict[str,pd.DataFrame]]='Golden'):
    from datavac.util.dvlogging import logger
    from datavac.config.data_definition import DDEF
    if (newer is None) or isinstance(newer,str):
        newer=maybe_newer['data'] if 'data' in (maybe_newer:=_open_rerun_data_file(newer)) else maybe_newer
    if isinstance(older,str):
        older=maybe_older['data'] if 'data' in (maybe_older:=_open_rerun_data_file(older)) else maybe_older
    found_problem=False
    assert isinstance(newer,dict)
    assert isinstance(older,dict) 
    for k in older:
        if k not in newer:
            oldtab=older[k]
            if len(oldtab)!=0:
                found_problem=True
                logger.critical(f"MISMATCH ERROR (Tab): Key {k} not found in newer data")
        else:
            newtab=newer[k]; oldtab=older[k];
            found_problem_in_this_tab=False
            if len(oldtab)==0:
                if len(newtab)>0:
                    logger.debug(f"                Note: Data has been added for {k}, which was previously empty")
                continue
            old_samples=list(oldtab[DDEF().sample_identifier_column.name].unique())
            new_samples=list(newtab[DDEF().sample_identifier_column.name].unique())
            if old_samples != new_samples:
                found_problem=True; found_problem_in_this_tab=True
                logger.critical(f"MISMATCH ERROR (Tab): Samples in key {k} do not match, "
                               f"old: {old_samples} vs new: {new_samples}")
            newtab=newtab[newtab[DDEF().sample_identifier_column.name].isin(old_samples)].copy()
            if len(oldtab) and (len(oldtab) != len(newtab)):
                found_problem=True; found_problem_in_this_tab=True
                logger.critical(f"MISMATCH ERROR (Tab): Length of data for key {k} does not match: "
                               f"{len(oldtab)} vs {len(newtab)}")
            else:
                # Temporary adjustments for old name scheme
                # TODO: remove when all data is migrated to new name scheme
                for tab in [oldtab, newtab]:
                    if k=='Probecheck Manipulator': 
                        1+1
                    if 'loadid' in tab.columns and 'measid' in tab.columns:
                        tab.sort_values(['loadid','measid'], inplace=True)
                        tab.reset_index(drop=True,inplace=True)
                    if 'matid' in tab.columns: tab.rename(columns={'matid':'sampleid'}, inplace=True)
                    if 'Mask' in tab.columns: tab.rename(columns={'Mask':'MaskSet'}, inplace=True)
                    tab.drop(columns=[c for c in tab.columns if (c.endswith('__1') and ('id' in c or 'Mask' in c))], inplace=True)
                    tab.drop(columns=[c for c in tab.columns if c in ['sampleid','loadid','MeasGroup','pol','layout_pol','anlsid']], inplace=True)
                for c in oldtab.columns:
                    if c=='date_user_changed': continue
                    if c not in newtab.columns:
                        found_problem=True; found_problem_in_this_tab=True
                        logger.critical(f"MISMATCH ERROR (Col): Column {c} not found in newer data for key {k}")
                    elif len(oldtab) and (not oldtab[c].equals(newtab[c])):
                            if 'loadid' in c or 'anlsid' in c or c=='sampleid':
                                logger.debug(f"                Warn: Column {c} does not match for key {k}")
                                             
                            else:
                                if pd.api.types.is_float_dtype(oldtab[c]) and pd.api.types.is_float_dtype(newtab[c]) and \
                                    np.allclose(oldtab[c].to_numpy(), newtab[c].to_numpy(), rtol=1e-5, equal_nan=True):
                                    logger.debug(f"                Note: Column {c} matches for {k} w/ relative tol 1e-5")
                                else:
                                    found_problem=True; found_problem_in_this_tab=True
                                    logger.critical(f"MISMATCH ERROR (Col): Column {c} does not match for key {k}")
                for c in newtab.columns:
                    if c not in oldtab.columns:
                        logger.debug(f"                Note: Column {c} has been added to newer data in key {k}")
            if not found_problem_in_this_tab:
                logger.debug(f"                Good: {k}")
    for k in newer:
        if k not in older:
            logger.debug(f"                Note: Key {k} has been added to newer data")

    if found_problem:
        raise Exception("Mismatch found in rerun data")
    else:
        logger.debug("No mismatches found in rerun data")


def cli_rerun_data(*args):
    """
    Command line interface for rerun_data
    """
    import argparse
    parser = argparse.ArgumentParser(description='Rerun data and compare to golden data')
    parser.add_argument('-dc','--dont-compare', action='store_true', help='Skip comparing data to golden')
    parser.add_argument('-cc','--current-context', action='store_true', help='Just upload to the current db instead of forcing local/regtest.'\
                                '  Not for regression testing, doesn\'t output a rerun dir, just a convenience for development.')
    parser.add_argument('-old','--older', type=str, default='Golden', help='Older rerun data file to compare to (default "Golden")')
    parser.add_argument('-jc','--just-compare', action='store_true', help='Just compare data, do not rerun')
    args = parser.parse_args(args)

    if not args.just_compare:
        if args.current_context:
            do_the_rerun_uploads()
        else:
            dat=rerun_data()
            if not args.dont_compare:
                compare_data(dat,older=args.older)
    else:
        compare_data(older=args.older)

#compare_data(rerun_data(),'Golden')

if __name__ == '__main__':
    import sys
    #cli_rerun_data(*sys.argv[1:])
    #rerun_data()
    compare_data(older='Norm')
    #compare_data(older='Norm')
    #compare_data(newer='Norm',older='Golden')