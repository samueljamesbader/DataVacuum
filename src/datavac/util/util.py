import argparse
import base64
import io
import importlib.resources as irsc
import secrets
import subprocess
from functools import partial
from importlib import import_module

from collections import deque
from contextlib import contextmanager, wraps
from pathlib import Path
from typing import Callable
import pickle


def last(it):
    return deque(it,maxlen=1).pop()
def first(it):
    return next(iter(it))
def only(seq,message=None):
    assert len(seq)==1, (message if message else f"This list should have exactly one element: {seq}.")
    return list(seq)[0]
def only_row(df,message=None):
    assert len(df)==1, (message if message else f"This table should have only one row \n{str(df)}")
    return df.iloc[0]


def run_subprocess_with_live_output(cmd_and_args, live_output=True, cwd=None):

    caught_output=io.StringIO()
    p = subprocess.Popen(cmd_and_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    for line in iter(lambda : p.stdout.readline(), b''):
        line=line.decode('utf-8')
        if live_output:
            print('>>>',line.strip(),flush=True)
        caught_output.write(line)
    p.stdout.close()
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
    pkg,relpath=dotpath.split(":")
    with irsc.as_file(irsc.files(pkg)) as pkg_path:
        return pkg_path/relpath


@contextmanager
def returner_context(val):
    yield val

def pickle_cached(cache_dir:Path, namer: Callable):
    """

    Example
    -------
        def expensive_getter(key,**kwargs):
            print(f"I'm expensive {key}")

        from datavac.util.paths import USER_CACHE
        CACHE=USER_CACHE/"example"
        cached_getter=pickle_cached(CACHE,lambda key,**kwargs: f"{key}.pkl")(expensive_getter)

    """
    def wrapper(func):
        if not cache_dir.exists(): cache_dir.mkdir()
        @wraps(func)
        def wrapped(*args,force=False,**kwargs):
            cfile=cache_dir/namer(*args,**kwargs)
            try:
                if not force:
                    with open(cfile,'rb') as f:
                        return pickle.load(f)
            except: pass
            res=func(*args,**kwargs)
            with open(cfile,'wb') as f:
                pickle.dump(res,f)
            return res
        return wrapped
    return wrapper

def base64encode(s):
    return base64.b64encode(s.encode()).decode()

# TODO: tuck these into a 'datavac util ...'
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