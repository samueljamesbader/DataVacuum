import sys
import argparse
import os
from pathlib import Path
from importlib import resources as irsc

import platformdirs
import yaml
from dotenv import load_dotenv

from datavac.util.cli import cli_helper
from datavac.util.util import import_modfunc

CONFIG: 'Config' = None

class Config():

    def __init__(self):
        with open(Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"project.yaml",'r') as f:
            self._yaml=yaml.safe_load(f)

    def __getattr__(self, item):
        return self._yaml[item]

    def __getitem__(self, item):
        return self._yaml[item]

    def get_meas_type(self, meas_group):
        res=self.measurement_groups[meas_group]['meas_type']
        if type(res) is str:
            return import_modfunc(res)()
        else:
            return import_modfunc(res[0])(**res[1])

    def get_dependent_analyses(self, meas_groups):
        return list(set(an for an,an_info in self.higher_analyses.items()\
            if any(mg in meas_groups for mg in
                   list(an_info.get('required_dependencies',{}).keys())+ \
                   list(an_info.get('attempt_dependencies',{}).keys()))))

    def get_dependency_meas_groups_for_analyses(self, analyses, required_only=False):
        if required_only:
            return dict(set([(mg,dname) for an in analyses
                 for mg,dname in self.higher_analyses[an]['required_dependencies'].items()]))
        else:
            return dict(set([(mg,dname) for an in analyses
                 for mg,dname in list(self.higher_analyses[an]['required_dependencies'].items())+\
                             list(self.higher_analyses[an].get('attempt_dependencies',{}).items())]))

    def get_dependency_meas_groups_for_meas_groups(self, meas_groups, required_only=False):
        if required_only:
            return dict(set([(mg,dname) for mg_ in meas_groups for mg,dname in
                             self.measurement_groups[mg_].get('required_dependencies',{}).items()]))
        else:
            return dict(set([(mg,dname) for mg_ in meas_groups for mg,dname in
                             list(self.measurement_groups[mg_].get('required_dependencies',{}).items())+ \
                             list(self.measurement_groups[mg_].get('attempt_dependencies',{}).items())]))

cli_context=cli_helper(cli_funcs={
    'list':'datavac.util.conf:cli_context_list',
    'use':'datavac.util.conf:cli_context_use',
})

CONTEXT_PATH: Path = None

def get_current_context_name():
    if (from_env:=os.environ.get('DATAVACUUM_CONTEXT',None)) is not None:
        assert (CONTEXT_PATH/f"{from_env}.dvcontext.env").exists(),\
            f"Context {from_env} (from DATAVACUUM_CONTEXT) not found in {CONTEXT_PATH}"
        return from_env
    elif (curr_file_path:=(CONTEXT_PATH/"current.txt")).exists():
        with open(curr_file_path) as f: context_name=f.read().strip()
        assert (CONTEXT_PATH/f"{context_name}.dvcontext.env").exists(),\
            f"Context {context_name} (named in {curr_file_path}) not found in {CONTEXT_PATH}"
    elif len(globs:=list(CONTEXT_PATH.glob("*.dvcontext.env")))==1:
        context_name=globs[0].name.split(".dvcontext.env")[0]
    elif len(globs)==0:
        #print(f"No context files found in {CONTEXT_PATH}")
        return None
    else:
        raise ValueError(f"Could not determine current context")
    os.environ['DATAVACUUM_CONTEXT']=context_name
    return context_name

def cli_context_list(*args):
    parser=argparse.ArgumentParser(description='Lists available contexts')
    namespace=parser.parse_args(args)
    ccn=get_current_context_name()
    for f in CONTEXT_PATH.glob("*.dvcontext.env"):
        context_name=f.name.split(".dvcontext.env")[0]
        print("*" if context_name==ccn else " ",context_name)

def cli_context_use(*args):
    parser=argparse.ArgumentParser(description='Selects the named context as current')
    parser.add_argument('context_name',help='The name of the context to use')
    namespace=parser.parse_args(args)
    assert (CONTEXT_PATH/f"{namespace.context_name}.dvcontext.env").exists(),\
        f"Context {namespace.context_name} not found in {CONTEXT_PATH}"
    with open(CONTEXT_PATH/"current.txt",'w') as f: f.write(namespace.context_name)

def load_config_pkg():
    conf_pkg=os.environ.get('DATAVACUUM_CONFIG_PKG',None)
    conf_dir=os.environ.get('DATAVACUUM_CONFIG_DIR',None)
    assert (conf_pkg is not None) or (conf_dir is not None), \
        "Must set one of DATAVACUUM_CONFIG_PKG or DATAVACUUM_CONFIG_DIR"
    if conf_pkg is not None:
        try:
            with irsc.as_file(irsc.files(conf_pkg)) as conf_path:
                assert conf_path.exists(), \
                    f"Config package {conf_pkg} not found at {str(conf_path)}"
                if conf_dir is not None:
                    assert conf_path.samefile(conf_dir), \
                        "DATAVACUUM_CONFIG_PKG and DATAVACUUM_CONFIG_DIR point to different files"
                os.environ['DATAVACUUM_CONFIG_DIR']=str(conf_path)
        except ModuleNotFoundError as e:
            raise Exception(f"Config package {conf_pkg} pointed to by DATAVACUUM_CONFIG_PKG not found") from e

def config_datavacuum():
    global CONFIG, CONTEXT_PATH
    # Load context
    CONTEXT_PATH=Path(os.environ.get('DATAVACUUM_CONTEXT_DIR',None)
         or platformdirs.user_config_path('ALL',appauthor='DataVacuum'))

    if Path(sys.argv[0]).stem=='datavac_with_context': return

    if CONTEXT_PATH=='None': print("Context-free")
    else:
        if not CONTEXT_PATH.exists(): CONTEXT_PATH.mkdir(parents=True,exist_ok=True)
        #IF it's a datavac-config
        context_name=get_current_context_name()
        if context_name is not None:
            load_dotenv(CONTEXT_PATH/f"{context_name}.dvcontext.env",override=True)

    # Load config
    #print(f"Using context {context_name}")
    load_config_pkg()
    CONFIG=Config()

def cli_datavac_with_context():
    import subprocess
    parser=argparse.ArgumentParser(description='Run a command with the context set')
    parser.add_argument('context_name',help='The name of the context to use')
    parser.add_argument('command',nargs='+',help='The command to run')
    namespace=parser.parse_args(sys.argv[1:3])

    assert (CONTEXT_PATH/f"{namespace.context_name}.dvcontext.env").exists(), \
        f"Context {namespace.context_name} not found in {CONTEXT_PATH}"

    env=os.environ.copy()
    env['DATAVACUUM_CONTEXT']=namespace.context_name
    command=sys.argv[2:]

    subprocess.run(command,env=env,shell=True,check=True,stdout=sys.stdout,stderr=sys.stderr,stdin=sys.stdin)