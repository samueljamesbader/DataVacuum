import sys
import argparse
import os
import warnings
from pathlib import Path
from importlib import resources as irsc

import platformdirs
import requests
import yaml
from dotenv import load_dotenv
from urllib3.exceptions import InsecureRequestWarning

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
    'install':'datavac.util.conf:cli_context_install',
    'edit': 'datavac.util.conf:cli_context_edit',
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


    if CONTEXT_PATH=='None': print("Context-free")
    else:
        if not CONTEXT_PATH.exists(): CONTEXT_PATH.mkdir(parents=True,exist_ok=True)

        if Path(sys.argv[0]).stem=='datavac_with_context': return # TODO: replace with 'datavac context with' and get rid of this
        if (Path(sys.argv[0]).stem=='datavac') and (len(sys.argv)>1 and sys.argv[1] in ['context','cn']): return

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

def cli_context_install(*args):
    parser=argparse.ArgumentParser(description='Install context from a deployment')
    parser.add_argument('url',help='The URL of the deployment')
    parser.add_argument('--cert',help='Path to SSL certificate')
    namespace=parser.parse_args(sys.argv[1:3])

    CONTEXT_PATH=Path(os.environ.get('DATAVACUUM_CONTEXT_DIR',None)
                      or platformdirs.user_config_path('ALL',appauthor='DataVacuum'))
    if namespace.cert:
        assert Path(namespace.cert).exists(), f"Cert file {namespace.cert} not found"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=InsecureRequestWarning)
        res=requests.get(namespace.url+"/context",verify=(namespace.cert or False))
    assert res.status_code==200, f"Failed to download context from {namespace.url}"

    filepath=CONTEXT_PATH/res.headers['Content-Disposition'].split("filename=")[1]
    with open(filepath,'wb') as f:
        f.write(res.content)
    cli_context_use(res.headers['Content-Disposition'].split("filename=")[1].split(".dvcontext.env")[0])
    print(f"Context installed to {filepath} and activated.")

def cli_context_edit(*args):
    parser=argparse.ArgumentParser(description='Edit the current context')
    parser.add_argument('--context',help='The context to edit')
    parser.add_argument('--variable',help='The variable to edit')
    parser.add_argument('--value',help='The value to set the variable to')
    parser.add_argument('--ask-user',action='store_true',help='Prompt the user for the value')
    parser.add_argument('--path',action='store_true',help='Check that the value exists as a path')
    namespace=parser.parse_args(args)

    context_file=CONTEXT_PATH/f"{namespace.context or get_current_context_name()}.dvcontext.env"
    assert context_file.exists(),\
        f"Context {namespace.context} not found in {CONTEXT_PATH}"

    if not namespace.variable:
        raise NotImplementedError("Opening editor not enabled yet, supply a --variable VAR argument")
        #os.system(f"notepad {CONTEXT_PATH/f'{namespace.context}.dvcontext.env'}")
        pass
    else:
        if namespace.value:
            assert not namespace.ask_user
            val=namespace.value
            if namespace.path:
                assert Path(val).exists(), f"Path '{val}' does not exist"
        else:
            assert namespace.ask_user
            if (current_val:=os.environ.get(namespace.variable)):
                print(f"Current value of {namespace.variable} is '{current_val}'")
                if input("Should it be changed? [Y/n] ") in ['N','n']:
                    return
            while True:
                val=input(f"What should {namespace.variable} be set to? ").strip()
                if len(val):
                    if namespace.path:
                        if Path(val).exists(): break
                        else: print(f"Path '{val}' does not exist")
                    else: break

        with open(context_file,'r') as f:
            context_contents=f.read().split("\n")
        preexists=False
        for i,line in enumerate(context_contents):
            if line.startswith(f"{namespace.variable}="):
                context_contents[i]=f"{namespace.variable}={val}"
                preexists=True
        if not preexists:
            context_contents.append(f"{namespace.variable}={val}")
        with open(context_file,'w') as f:
            f.write("\n".join(context_contents))