import time
import logging
import sys
from contextlib import contextmanager


def _setup_logging():
    global logger, _ch
    logger = logging.getLogger('datavac')
    logger.setLevel(logging.DEBUG)
    for hndlr in list(logger.handlers):
        logger.removeHandler(hndlr)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.DEBUG)
    _ch.setFormatter(logging.Formatter('{asctime}.{msecs:03.0f}:: {message}',style='{',datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(_ch)

def set_level(level):
    if type(level) is str:
        level=getattr(logging,level.upper())
    _ch.setLevel(level)

@contextmanager
def level_context(level):
    _level=_ch.level
    set_level(level)
    try:
        yield
    finally:
        set_level(_level)

@contextmanager
def time_it(message, threshold_time:float=0):
    start_time=time.time()
    yield
    took_time=time.time()-start_time
    if took_time>threshold_time:
        logger.debug(f"{message} took {took_time:.5g}s")

#@contextmanager
#def log_context(message,level=logging.DEBUG):

logger: logging.Logger = None
_ch: logging.StreamHandler = None
_indent=''
_setup_logging()
