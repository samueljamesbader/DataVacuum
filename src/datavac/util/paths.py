import os
from pathlib import Path
from typing import Optional

import platformdirs

DEPLOYMENT_NAME: Optional[str] = os.environ.get('DATAVACUUM_DEPLOYMENT_NAME',None)
appname=DEPLOYMENT_NAME or 'DEFAULT'
USER_CACHE: Path = (Path(os.environ.get("DATAVACUUM_CACHE_DIR",None)) or \
                   platformdirs.user_cache_path(appname=appname, appauthor='DataVacuum'))/appname
USER_CERTS: Path = USER_CACHE/"certs"
USER_DOWNLOADS: Path = platformdirs.user_downloads_path()

USER_CACHE.mkdir(parents=True,exist_ok=True)
USER_CERTS.mkdir(parents=True,exist_ok=True)