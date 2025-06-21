import argparse
import base64
import io
import importlib.resources as irsc
import secrets
import subprocess
from functools import partial
from importlib import import_module

from collections import deque
from contextlib import contextmanager
from typing import Any


def last(it):
    return deque(it,maxlen=1).pop()
def first(it):
    return next(iter(it))
def only(seq,message=None):
    assert len(seq)==1, (message.replace('{seq}',str(seq)) if message else f"This list should have exactly one element: {seq}.")
    return list(seq)[0]
def only_row(df,message=None):
    assert len(df)==1, (message if message else f"This table should have only one row \n{str(df)}")
    return df.iloc[0]


def run_subprocess_with_live_output(cmd_and_args, live_output=True, cwd=None):

    caught_output=io.StringIO()
    p = subprocess.Popen(cmd_and_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    for line in iter(lambda : p.stdout.readline(), b''): # type: ignore
        line=line.decode('utf-8')
        if live_output:
            print('>>>',line.strip(),flush=True)
        caught_output.write(line)
    p.stdout.close() # type: ignore
    p.wait()
    caught_output.seek(0)
    return caught_output.read(), p.returncode

def import_modfunc(dotpath):
    if type(dotpath) is not str:
        dotpath,kwargs=dotpath
        assert type(kwargs) is dict
    else: kwargs=None
    try: mod,func=dotpath.split(':')
    except ValueError:
        raise ValueError(f"Dotpath '{dotpath}' is improperly formatted ('package.module:function')")
    mod=import_module(mod)
    func=getattr(mod,func)
    if kwargs: func=partial(func,**kwargs)
    return func

def get_resource_path(dotpath):
    pkg,relpath=dotpath.split("::")
    with irsc.as_file(irsc.files(pkg)) as pkg_path:
        return pkg_path/relpath


@contextmanager
def returner_context(val):
    yield val


def base64encode(s):
    return base64.b64encode(s.encode()).decode()

def cli_base64encode(*args):
    parser=argparse.ArgumentParser(description='Encodes a string to base64')
    parser.add_argument('thestring',nargs='?',help='the string to encode')
    namespace=parser.parse_args(args)
    thestring=namespace.thestring or input("String to encode: ")
    print(base64encode(thestring))

def cli_b64rand(*args):
    parser=argparse.ArgumentParser(description='Generates a cryptographically random 32 bytes and encodes it to base64')
    namespace=parser.parse_args(args)
    random_bytes = secrets.token_bytes(32)
    print(base64.b64encode(random_bytes).decode())


def asnamedict(*dvcol_list: Any) -> dict[str, Any]:
    """Converts a list of objects to a dictionary with object.name as keys.

    Example:
        >>> asnamedict([DVColumn('col1', 'int32', 'description1'),
                        DVColumn('col2', 'str',   'description2')])
        {'col1': DVColumn('col1', 'int32', 'description1'),
         'col2': DVColumn('col2', 'str',   'description2')}
    
    """
    return {col.name: col for col in dvcol_list}