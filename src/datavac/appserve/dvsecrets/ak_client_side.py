from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable, Optional
from datavac.config.project_config import PCONF
from datavac.util.dvlogging import logger


from datavac.util.util import only
if TYPE_CHECKING:
    import webbrowser

@contextmanager
def prepare_to_receive_key(callback: Callable):
    import http.server
    receiver_port = 8099
    class APIKeyReceiver(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                from urllib.parse import urlparse, parse_qs
                callback(only(parse_qs(urlparse(self.path).query)['message']))
            except Exception as e:
                self.send_response(403)
                self.end_headers()
                import traceback
                self.wfile.write(f"Error processing request: {traceback.format_exception(e)}".encode('utf-8'))
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write("""
                    <HTML><HEAD><script language="javascript" type="text/javascript">
                    function closeWindow() {
                    window.open('','_parent','');
                    window.close();
                    }
                    function delay() {
                    setTimeout("closeWindow()", 1500);
                    }
                    </script></HEAD>
                    <BODY onload="javascript:delay();">Successly received DataVacuum access key, closing window.</BODY></HTML>""".encode('utf-8'))
        def log_message(self, format: str, *args) -> None: pass
    server = http.server.HTTPServer(('localhost', receiver_port), APIKeyReceiver)
    logger.debug(f"JWT receiver listening on port {receiver_port}")
    import threading
    thread=threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield receiver_port
    finally:
        threading.Thread(target=server.shutdown).start()
        logger.debug("Server shut down.")


class AccessKeyRetrievalException(Exception):
    """Exception raised when the access key retrieval fails."""
    pass

def direct_user_to_access_key(specific_webbrowser:Optional[webbrowser.BaseBrowser]=None,
                              login_timeout:int=30, logout_first:bool=False) -> str:
    # Open a browser to the access key page
    if specific_webbrowser is None:
        import webbrowser
        specific_webbrowser = webbrowser.get()
    PCONF()
    from nacl.public import PublicKey, PrivateKey, SealedBox
    import base64
    skey=PrivateKey.generate()
    pkey=skey.public_key
    got_something=False
    jwt = None
    def save_key(message: str):
        nonlocal got_something, jwt
        got_something=True
        try:
            jwt=SealedBox(skey).decrypt(base64.urlsafe_b64decode(message.encode('utf-8'))).decode('utf-8')
        except Exception as e:
            import traceback
            logger.warning(f"Failed to decrypt message: {traceback.format_exception(e)}")
            raise e
        else:
            logger.debug(f"Received API key")
    with prepare_to_receive_key(save_key) as receiver_port:
        specific_webbrowser.open(PCONF().deployment_uri\
                        +"/accesskey"+\
                        f"?redirect_port={receiver_port}"\
                        f"&pkey={base64.urlsafe_b64encode(pkey.encode()).decode('utf-8')}"\
                        f"{'&logoutfirst=1' if logout_first else ''}"
                        ,new=1)
        poll= .1
        while not got_something and login_timeout > 0:
            login_timeout -= poll # type: ignore
            time.sleep(poll)
        if not got_something:
            raise AccessKeyRetrievalException("Failed to get API key from user (likely login failure/timeout)")
    if not jwt:
        raise AccessKeyRetrievalException("Failed to get API key from user")
    return jwt


_access_key_cache: Optional[str] = None
def _access_key_path() -> Path:
    cache = PCONF().USER_CACHE / "certs"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "datavacuum_access_key.jwt"
def get_access_key(specific_webbrowser:Optional[webbrowser.BaseBrowser]=None, login_timeout:int=30,
                   if_directing_logout_first:bool=False) -> str:
    global _access_key_cache
    if _access_key_cache is not None:
        return _access_key_cache
    if (akp:=_access_key_path()).exists():
        with open(akp, 'r') as f:
            _access_key_cache = f.read()
    else:
        jwt=direct_user_to_access_key(specific_webbrowser=specific_webbrowser, login_timeout=login_timeout, logout_first=if_directing_logout_first)
        with open(akp, 'w') as f:
            f.write(jwt)
        _access_key_cache = jwt
    return _access_key_cache
def invalidate_access_key():
    global _access_key_cache
    _access_key_cache = None
    _access_key_path().unlink(missing_ok=True)

def refresh_access_key(logout_first:bool=False):
    invalidate_access_key()
    return get_access_key(if_directing_logout_first=logout_first)

def cli_refresh_access_key(*args):
    parser=argparse.ArgumentParser(prog='refresh_access_key',
                                   description='Refresh the cached access key by logging in again')
    parser.add_argument('--logout_first','-lo',action='store_true',
                        help='Clear cookies in the browser first, to ensure a fresh login')
    namespace=parser.parse_args(args)
    refresh_access_key(logout_first=namespace.logout_first)
    return cli_print_user()

def store_demo_access_key(user:str, roles:list[str], other_info={}):
    try: akss=PCONF().vault.get_access_key_sign_seed()
    except:
        logger.critical("Couldn't get signing seed for access key.")
        raise
    try: aud=PCONF().vault.get_auth_info().get('oauth_key',None)
    except:
        logger.critical("Couldn't get audience (oauth_key) for access key.")
        raise

    import jwt
    other_info=other_info.copy()
    if 'iat' not in other_info: other_info['iat']=time.time()
    jwt_token = jwt.encode({'unique_name': user, 'roles': roles, 'aud':aud, **other_info}, akss, algorithm='HS256')

    with open(_access_key_path(), 'w') as f:
        f.write(jwt_token)
    return jwt_token

class AccessKeyError(PermissionError): pass
class AccessKeyExpiredError(AccessKeyError):
    """Exception raised when the access key has expired."""
    pass
class AccessKeyInvalidError(AccessKeyError):
    """Exception raised when there's some other issue authenticating to the server by access key."""
    pass
class AccessKeyPermissionError(AccessKeyError):
    """Exception raised when the user does not have permission for the requested resource by access key."""
    pass

from datavac.appserve.api import client_server_split
@client_server_split(method_name='get_user',return_type='ast',split_on='is_server')
def get_user(user_info:dict={}):
    return {k:user_info[k] for k in ['unique_name','roles','iat'] if k in user_info}
def cli_print_user(*args):
    import argparse
    parser=argparse.ArgumentParser(prog='print_user', description='Print the user info from the access key')
    namespace=parser.parse_args(args)
    ui=get_user()
    print("\nUser info from access key")
    print("--------------------------------")
    for k,v in ui.items():
        print(f"{k}: {v}")
    if 'iat' in ui:
        print(f"\nAccess key issued at {datetime.fromtimestamp(ui['iat'])} (age {datetime.now()-datetime.fromtimestamp(ui['iat'])})\n")
    return ui

def cli_invalidate_access_key(*args):
    import argparse
    parser=argparse.ArgumentParser(prog='invalidate_access_key', description='Invalidate the cached access key')
    namespace=parser.parse_args(args)
    invalidate_access_key()
    print("Invalidated cached access key.")