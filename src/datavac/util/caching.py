from __future__ import annotations
import argparse
from datetime import datetime
import functools
import pickle
import shutil
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from datavac.util.logging import time_it, logger

if TYPE_CHECKING:
    from sqlalchemy import Connection


def pickle_cached(cache_dir:str|Path, # type: ignore
                  namer: Callable[..., str])\
            -> Callable[[Callable], Callable]:
    """
    A decorator to cache the results of a function using pickle.

    Args:
        cache_dir (str | Path): The directory where the cache files will be stored.
            If not absolute, will be interpreted relative to PCONF().USER_CACHE.
        namer (Callable[[Any], str], optional): A function that takes the wrapped-function's
            arguments and returns a string to use as the cache file name.

    Example
    -------
        >>> def expensive_getter(key,**kwargs):
        >>>    print(f"I'm expensive {key}")
        >>>    return f"Here's {key}!"
        >>> cached_getter=pickle_cached("example_dir",namer=lambda x: f"{x}.pkl")(expensive_getter)
        >>> cached_getter("test_key")  
        I'm expensive test_key
        Here's test_key!
        >>> cached_getter("test_key")  
        Here's test_key!

        This expensive result is cached in PCONF().USER_CACHE/'example_dir/test_key.pkl'.

    """
    cache_dir:Path=Path(cache_dir)
    def wrapper(func):
        nonlocal cache_dir

        # Ensure the cache directory is absolute and exists
        # but don't do so until the first call to the wrapped function
        # This allows the cache directory to be set up only when needed
        # and, crucially, prevents accessing PCONF() until runtime
        # so pickle_cached can be used in modules that are defined
        # before the project config is set up.
        is_setup = False
        def setup():
            nonlocal is_setup, cache_dir
            if not cache_dir.is_absolute():
                from datavac.config.project_config import PCONF
                cache_dir = PCONF().USER_CACHE/cache_dir
            assert 'cache' in str(cache_dir).lower(), \
                f"Cache directory {cache_dir} should contain 'cache' in its name"\
                    " to avoid accidental overwrite of important directories."
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)

        # The actual wrapper just caching via a pickle in the above-setup-up directory 
        @wraps(func)
        def wrapped(*args,force=False,**kwargs):
            if not is_setup: setup()
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


def pickle_db_cached(namer: Union[Callable[[],str],str], namespace:str, conn:Optional[Connection]=None):
    def wrapper(func):
        from datavac.config.project_config import PCONF
        cache_dir = PCONF().USER_CACHE/namespace
        if not cache_dir.exists(): cache_dir.mkdir()
        from datavac.database import blob_store 

        @functools.wraps(func)
        def wrapped(*args,force=False,**kwargs):
            name: str =namer if type(namer) is str else namer(*args,**kwargs) # type: ignore
            cfile=cache_dir/name
            blob_name=f'{namespace}.{name}'
            import numpy as np
            if not force:
                from sqlalchemy.exc import SQLAlchemyError
                with time_it(f"Get DB cache time for {name}",threshold_time=.005):
                    try: db_cached_time:float= blob_store.get_obj_date(blob_name,conn=conn).timestamp()
                    # If it's an SQLAlchemyError, the transaction might be aborted so have to raise this
                    # Silently aborted transations could cause some downstream use of this connection to fail
                    except SQLAlchemyError as e: raise e
                    # Otherwise, we'll just try to fix it
                    except Exception as e:
                        logger.debug("Couldn't get DB cache time: "+str(e));
                        db_cached_time:float= -np.inf
                with time_it(f"Get local cache time for {name}",threshold_time=.005):
                    try: lc_cached_time=cfile.stat().st_mtime
                    except Exception as e:
                        logger.debug("Couldn't get local cache time: "+str(e));
                        lc_cached_time:float= -np.inf
                if (not np.isfinite(db_cached_time)) and (not np.isfinite(lc_cached_time)):
                    logger.debug(f"Couldn't get local or DB cache time for {name}")
                    force=True
            if not force:
                if db_cached_time<lc_cached_time: # type: ignore
                    try:
                        with time_it(f"Reading local cache for {name}",threshold_time=.005):
                            with open(cfile,'rb') as f: return pickle.load(f),lc_cached_time
                    except Exception as e:
                        logger.debug(f"Couldn't read local cache for {name}: {str(e)}")
                try:
                    with time_it(f"Reading DB cache for {name}",threshold_time=.025):
                        res,cached_time=blob_store.get_obj(f'{namespace}.{name}',conn=conn),db_cached_time
                except Exception as e:
                    logger.debug(f"Couldn't read DB cache for {name}: {str(e)}")
                    force=True
                else:
                    with time_it(f"Storing local cache for {name}",threshold_time=.005):
                        with open(cfile,'wb') as f: pickle.dump(res,f)
                    return res,cached_time
            with time_it(f"Generating {name}"):
                res=func(*args,**kwargs)

            with time_it(f"Storing DB cache for {name}",threshold_time=.025):
                blob_store.store_obj(blob_name,res,conn=conn)
            with time_it(f"Storing local cache for {name}",threshold_time=.005):
                with open(cfile,'wb') as f: pickle.dump(res,f)
            return res,datetime.now().timestamp()
        return wrapped
    return wrapper


def clear_local_cache(cache_dir: Optional[str | Path]):
    """
    Clears the cache directory by removing it and creating a new empty one.
    
    Parameters
    ----------
    cache_dir : str | Path, optional
        The path to the cache directory to clear. Defaults to 'cache'.
    """
    if cache_dir is None:
        from datavac.config.project_config import PCONF
        cache_dir = PCONF().USER_CACHE
    else:
        cache_dir = Path(cache_dir)
        if not cache_dir.is_absolute():
            from datavac.config.project_config import PCONF
            cache_dir = PCONF().USER_CACHE / cache_dir
    assert 'cache' in str(cache_dir).lower(), \
        f"Cache directory {cache_dir} should contain 'cache' in its name"\
            " to avoid accidental deletion of important directories."
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir()

def cli_clear_local_cache(*args):
    parser=argparse.ArgumentParser(description='Clears the user cache')
    parser.add_argument(
        '--cache-dir', type=str, default=None,
        help='The path to the cache directory to clear. Defaults to the user cache directory.'
    )
    namespace=parser.parse_args(args)
    clear_local_cache(namespace.cache_dir)
