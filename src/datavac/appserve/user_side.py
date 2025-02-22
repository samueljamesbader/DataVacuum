import requests

from datavac.appserve.dvsecrets import get_ssl_rootcert_for_ak
from datavac.util.logging import logger, time_it
import webbrowser
import os
from pathlib import Path
import shutil
from datavac.util.paths import USER_CERTS, USER_DOWNLOADS



def direct_user_to_access_key():
    # Open a browser to the access key page
    webbrowser.open(os.environ['DATAVACUUM_DEPLOYMENT_URI']+"/accesskey",new=1)
def copy_in_access_key():
    possibles=list(USER_DOWNLOADS.glob("datavacuum_access_key*.txt"))
    if not len(possibles):
        logger.debug("No access key in Downloads")
    else:
        selected=max(possibles,key=lambda filename:Path(filename).lstat().st_mtime)
        shutil.copy(selected,USER_CERTS/"datavacuum_access_key.txt")
        for f in possibles:
            os.remove(f)
        logger.debug(f"Copied over {str(selected)} to cache")

def have_user_download_access_key():
    input("When you hit enter, a browser window will open for you to download an access key.\n" 
        "Let this key download to your default Downloads folder, then close that browser window.")
    direct_user_to_access_key()
    input('Press enter once downloaded')
    copy_in_access_key()


def get_saved_access_key(suppress_error=False):
    try:
        with open(USER_CERTS/"datavacuum_access_key.txt",'rb') as f:
            return f.read().decode()
    except FileNotFoundError:
        if not suppress_error:
            raise
        return None

def is_access_key_valid():
    assert get_saved_access_key() is not None
    try: get_secret_from_deployment('readonly_dbstring')
    except Exception as e:
        if ('not valid' in str(e)) or ('expired' in str(e)):
            return False
        else: raise
    return True

def get_secret_from_deployment(secret_name):
    try: access_key=get_saved_access_key()
    except:
        if (os.environ.get('DATAVACUUM_FROM_JMP')=='YES'):
            raise Exception("No access key")
        have_user_download_access_key()
        access_key=get_saved_access_key()
    with time_it("Secret-share request"):
        response=requests.post(os.environ['DATAVACUUM_DEPLOYMENT_URI'] +"/secretshare",
                               data={"secretname":secret_name,"access_key":access_key},
                               verify=str(get_ssl_rootcert_for_ak()))
    assert response.status_code==200, f"Failed to get {secret_name} from deployment: {response.text}"
    return response.text