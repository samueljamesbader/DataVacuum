from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable, Optional
from datavac.config.project_config import PCONF
from datavac.util.dvlogging import logger

import http.server
import threading

from datavac.util.util import only
if TYPE_CHECKING:
    import webbrowser

@contextmanager
def prepare_to_receive_key(callback: Callable):
    receiver_port = 3001
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
                    setTimeout("closeWindow()", 1000);
                    }
                    </script></HEAD>
                    <BODY onload="javascript:delay();">Success, closing window.</BODY></HTML>""".encode('utf-8'))
        def log_message(self, format: str, *args) -> None: pass
    server = http.server.HTTPServer(('localhost', receiver_port), APIKeyReceiver)
    logger.debug(f"JWT receiver listening on port {receiver_port}")
    thread=threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield receiver_port
    finally:
        threading.Thread(target=server.shutdown).start()
        logger.debug("Server shut down.")


class AccessKeyRetrievalException(Exception):
    """Exception raised when the access key retrieval fails."""
    pass

def direct_user_to_access_key(specific_webbrowser:Optional[webbrowser.BaseBrowser]=None, login_timeout:int=30) -> str:
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
def get_access_key(specific_webbrowser:Optional[webbrowser.BaseBrowser]=None, login_timeout:int=30) -> str:
    global _access_key_cache
    if _access_key_cache is not None:
        return _access_key_cache
    if (akp:=_access_key_path()).exists():
        with open(akp, 'r') as f:
            _access_key_cache = f.read()
    else:
        jwt=direct_user_to_access_key(specific_webbrowser=specific_webbrowser, login_timeout=login_timeout)
        with open(akp, 'w') as f:
            f.write(jwt)
        _access_key_cache = jwt
    return _access_key_cache
def invalidate_access_key():
    global _access_key_cache
    _access_key_cache = None
    _access_key_path().unlink(missing_ok=True)

