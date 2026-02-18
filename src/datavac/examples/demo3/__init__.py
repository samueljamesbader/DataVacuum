import os
from pathlib import Path
from datavac.config.contexts import get_current_context_name

assert get_current_context_name()=='builtin:demo3',\
    "To use demo3 content, switch to the builtin:demo3 context:\n"\
    "> datavac context use builtin:demo3"
dbname='datavacuum_demo3'
EXAMPLE_DATA_DIR = Path(os.environ['DATAVACUUM_TEST_DATA_DIR']) / 'demo3'