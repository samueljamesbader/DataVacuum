import os
from pathlib import Path
import platformdirs


assert os.environ['DATAVACUUM_CONTEXT']=='builtin:demo1'

READ_DIR=Path(os.environ.get('DATAVACUUM_TEST_DATA_DIR',
                             platformdirs.user_cache_path('ALL',appauthor='DataVacuum')/'example_data'))
os.environ['DATAVACUUM_DEPLOYMENT_NAME']='demo1'
os.environ['DATAVACUUM_READ_DIR']=str(READ_DIR)
os.environ['DATAVACUUM_LAYOUT_PARAMS_DIR']=str(Path(__file__).parent/"config/layout_params")

# Set the environment variables for a local test database
os.environ['DATAVACUUM_DBSTRING']=f"Server = localhost; Port = 5432; Database = datavacuum_test;" \
                                  f" Uid = postgres; Password = {os.environ.get('DATAVACUUM_TEST_DB_PASS','')}"
os.environ['DATAVACUUM_DB_DRIVERNAME']="postgresql"
