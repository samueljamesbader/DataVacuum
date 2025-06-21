import datetime
import pickle
from typing import Union, Optional

import pandas as pd
import yaml
import os
from pathlib import Path

from datavac.io.database import get_database, read_and_upload_data
from datavac.tests.freshtestdb import make_fresh_testdb
from datavac.util.conf import CONFIG
from datavac.util.logging import logger

yaml_path = Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"rerun_data.yaml"
dbname='regtest'
os.environ['DATAVACUUM_DBSTRING']=f"Server = localhost; Port = 5432; Database = {dbname};" \
                                  f" Uid = postgres; Password = {os.environ.get('DATAVACUUM_TEST_DB_PASS','')}"
os.environ['DATAVACUUM_DB_DRIVERNAME']="postgresql"
RERUN_DIR=Path(os.environ["DATAVACUUM_RERUN_DIR"])
timefmt="%Y-%m-%d_%H-%M-%S"

def rerun_data():
    """
    Rerun the data in the rerun_data.yaml file
    """
    assert yaml_path.exists(), f"rerun_data.yaml file not found at {yaml_path}"
    with open(yaml_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    make_fresh_testdb(dbname)
    db=get_database()
    for ud in yaml_dict['regression_data']['uploads']:
        read_and_upload_data(db,**ud)
    RERUN_DIR.mkdir(exist_ok=True,parents=False)
    destfile=RERUN_DIR/(datetime.datetime.now().strftime(timefmt)+".pkl")
    dat={mg:db.get_data(mg) for mg in list(CONFIG['measurement_groups'])+list(CONFIG['higher_analyses'])}
    with open(destfile, 'wb') as f:
        pickle.dump(dat,f)
    logger.debug(f"Data rerun and saved to {destfile}")
    return dat

def compare_data(newer: Optional[Union[str,dict[str,pd.DataFrame]]]=None,
                 older: Union[str,dict[str,pd.DataFrame]]='Golden'):
    if newer is None:
        potential_files=list(RERUN_DIR.glob('2*.pkl'))
        if len(potential_files) == 0:
            raise ValueError(f"No rerun data files found in {RERUN_DIR}")
        newer = max(potential_files, key=lambda file: datetime.datetime.strptime(file.stem, timefmt)).stem
        logger.debug(f"Using most recent rerun data file: {newer}.pkl")
    if (type(newer) is str) or isinstance(newer,Path):
        with open(RERUN_DIR/f"{newer}.pkl", 'rb') as f:
            newer=pickle.load(f)
    if (type(older) is str) or isinstance(older,Path):
        with open(RERUN_DIR/f"{older}.pkl", 'rb') as f:
            older=pickle.load(f)
    found_problem=False
    for k in older:
        if k not in newer:
            found_problem=True
            logger.critical(f"MISMATCH ERROR: Key {k} not found in newer data")
        else:
            newtab=newer[k]; oldtab=older[k];
            if len(oldtab)==0:
                if len(newtab)>0:
                    logger.debug(f"Data has been added for {k}, which was previously empty")
            for c in oldtab.columns:
                if c=='date_user_changed': continue
                if c not in newtab.columns:
                    found_problem=True
                    logger.critical(f"MISMATCH ERROR: Column {c} not found in newer data for key {k}")
                elif len(oldtab) and (not oldtab[c].equals(newtab[c])):
                        found_problem=True
                        logger.critical(f"MISMATCH ERROR: Column {c} does not match for key {k}")
            for c in newtab.columns:
                if c not in oldtab.columns:
                    logger.debug(f"Column {c} has been added to newer data in key {k}")
    for k in newer:
        if k not in older:
            logger.debug(f"Key {k} has been added to newer data")

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
    args = parser.parse_args(args)

    dat=rerun_data()
    if not args.dont_compare:
        compare_data(dat,'Golden')

#compare_data(rerun_data(),'Golden')
#compare_data()
