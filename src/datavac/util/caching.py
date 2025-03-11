import argparse
import pickle
import shutil
from functools import wraps
from pathlib import Path
from typing import Callable

from datavac.util.paths import USER_CACHE


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


def cli_clear_cache(*args):
    parser=argparse.ArgumentParser(description='Clears the user cache')
    namespace=parser.parse_args(args)
    shutil.rmtree(USER_CACHE)
    USER_CACHE.mkdir()
