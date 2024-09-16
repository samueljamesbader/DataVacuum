import io
import subprocess
from functools import partial
from importlib import import_module

from collections import deque
from contextlib import contextmanager


def last(it):
    return deque(it,maxlen=1).pop()
def first(it):
    return next(iter(it))
def only(seq,message=None):
    assert len(seq)==1, (message if message else f"This list should have exactly one element: {seq}.")
    return seq[0]
def only_row(df,message=None):
    assert len(df)==1, (message if message else f"This table should have only one row {str(df)}")
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

@contextmanager
def returner_context(val):
    yield val



