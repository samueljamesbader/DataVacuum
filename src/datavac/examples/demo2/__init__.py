import os
from pathlib import Path
from datavac.config.contexts import get_current_context_name

assert get_current_context_name()=='builtin:demo2',\
    "To use demo2 content, switch to the builtin:demo2 context:\n"\
    "> datavac context use builtin:demo2"
dbname='datavacuum_demo2'
EXAMPLE_DATA_DIR = Path(os.environ['DATAVACUUM_TEST_DATA_DIR']) / 'demo2'