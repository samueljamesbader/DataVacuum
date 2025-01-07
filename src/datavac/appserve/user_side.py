from datavac.util.logging import logger
import webbrowser
import os
from pathlib import Path
import shutil
from datavac.util.paths import USER_CERTS, USER_DOWNLOADS


def have_user_download_access_key():
    # Open a browser to the access key page
    webbrowser.open(os.environ['DATAVACUUM_DEPLOYMENT_URI']+"/accesskey",new=1)
    input('Press enter once downloaded')

    possibles=list(USER_DOWNLOADS.glob("datavacuum_access_key*.txt"))
    if not len(possibles):
        logger.debug("No access key in Downloads")
    else:
        selected=max(possibles,key=lambda filename:Path(filename).lstat().st_mtime)
        shutil.copy(selected,USER_CERTS/"datavacuum_access_key.txt")
        for f in possibles:
            os.remove(f)
        logger.debug(f"Copied over {str(selected)} to cache")

def get_saved_access_key():
    with open(USER_CERTS/"datavacuum_access_key.txt",'rb') as f:
        return f.read()

