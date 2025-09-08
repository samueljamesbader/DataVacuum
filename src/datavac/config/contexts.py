from __future__ import annotations
from typing import TYPE_CHECKING
import argparse
from functools import cache
import os
from pathlib import Path
import sys
import warnings

from datavac.util.cli import CLIIndex
from dotenv import load_dotenv
import platformdirs

if TYPE_CHECKING:
    from tornado.web import RequestHandler

@cache
def get_current_context_name():
    if (from_env:=os.environ.get('DATAVACUUM_CONTEXT',None)) is not None:
        if from_env.startswith('builtin:'): return from_env
        else:
            assert (CONTEXT_PATH/f"{from_env}.dvcontext.env").exists(),\
                f"Context {from_env} (from DATAVACUUM_CONTEXT) not found in {CONTEXT_PATH}"
            return from_env
    elif (curr_file_path:=(CONTEXT_PATH/"current.txt")).exists():
        with open(curr_file_path) as f: context_name=f.read().strip()
        if context_name.startswith('builtin:'): return context_name
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


def load_environment_from_context():
    global CONFIG, CONTEXT_PATH
    def _use_builtin_context(biname:str):
        os.environ['DATAVACUUM_CONFIG_MODULE']=f'datavac.examples.{biname}.{biname}_dvconfig'
        if 'DATAVACUUM_CONFIG_PATH' in os.environ: del os.environ['DATAVACUUM_CONFIG_PATH']
        os.environ['DATAVACUUM_DIRECT_DB_ACCESS']='YES'
        os.environ['DATAVACUUM_IS_SERVER']='YES'

    if (envcont:=os.environ.get('DATAVACUUM_CONTEXT','')).startswith('builtin:'):
        _use_builtin_context(envcont.split(":")[1])
    else:

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
                os.environ['DATAVACUUM_CONTEXT']=context_name
                if context_name.startswith('builtin:'):
                    _use_builtin_context(context_name.split(":")[1])
                else:
                    load_dotenv(CONTEXT_PATH/f"{context_name}.dvcontext.env",override=True)

def cli_context_list(*args):
    parser=argparse.ArgumentParser(description='Lists available contexts')
    namespace=parser.parse_args(args)
    ccn=get_current_context_name()
    print(f"Current context: \"{ccn}\"")
    print("Available contexts (excluding builtins):")
    for f in CONTEXT_PATH.glob("*.dvcontext.env"):
        context_name=f.name.split(".dvcontext.env")[0]
        print("*" if context_name==ccn else " ",context_name)


def cli_context_use(*args):
    parser=argparse.ArgumentParser(description='Selects the named context as current')
    parser.add_argument('context_name',help='The name of the context to use')
    namespace=parser.parse_args(args)
    if not namespace.context_name.startswith('builtin:'):
        assert (CONTEXT_PATH/f"{namespace.context_name}.dvcontext.env").exists(),\
            f"Context {namespace.context_name} not found in {CONTEXT_PATH}"
    with open(CONTEXT_PATH/"current.txt",'w') as f: f.write(namespace.context_name)



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

    import requests
    from urllib3.exceptions import InsecureRequestWarning
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

def get_context_download_request_handler() -> type[RequestHandler]:
    import datetime
    from tornado.web import RequestHandler
    class ContextDownload(RequestHandler):
        def get(self):
            from datavac.config.project_config import PCONF
            depname=PCONF().deployment_name
            self.set_header('Content-Disposition', f'attachment; filename={depname}.dvcontext.env')
            self.write(f"# Context file for '{depname}'\n")
            self.write(f"# Downloaded {datetime.datetime.now()}\n")
            for name,val in [('DATAVACUUM_DEPLOYMENT_NAME',depname),
                             ('DATAVACUUM_DEPLOYMENT_URI',PCONF().deployment_uri),
                             ('DATAVACUUM_CONFIG_MODULE', os.environ['DATAVACUUM_CONFIG_MODULE'])]:
                self.write(f"{name}={val}\n")
    return ContextDownload

CONTEXT_CLI = CLIIndex({
    'list':cli_context_list,
    'use':cli_context_use,
    'install':cli_context_install,
    'edit': cli_context_edit,
})