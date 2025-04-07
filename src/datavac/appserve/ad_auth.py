import base64
import shutil

import yaml
import os
from datetime import datetime, timedelta
from typing import Callable

import nacl.signing
import panel
from nacl.encoding import Base64Encoder
from nacl.exceptions import BadSignatureError
from panel.auth import AzureAdLoginHandler, OAuthProvider

from datavac.appserve.dvsecrets import get_access_key_sign_seed, get_auth_info
from datavac.util.logging import logger

# TODO: Check if any recent Panel updates provide a way to not require this anymore

class AzureAdLogoutHandler(AzureAdLoginHandler):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    async def get(self,*args,**kwargs):
        logger.debug(f"Logging user out")
        self.set_header("Clear-Site-Data", '"cache","cookies","storage"')
        #self.redirect("/")
        self.write("Your cookies should be clear!")

def monkeypatch_oauthprovider():
    OAuthProvider.logout_handler = property(lambda self: AzureAdLogoutHandler)
    logger.debug("Oauth provider monkey-patched to support logout")





from panel.io.server import fullpath, COMPONENT_PATH, ComponentResourceHandler
from tornado.web import StaticFileHandler, authenticated
from panel.config import config
class AuthedStaticFileHandler(StaticFileHandler):

    def initialize(self,path:str,role:str,*args,**kwargs):
        self.require_auth=(get_auth_info()['oauth_provider']!="none")
        assert role is not None
        self.required_role:str=role
        super().initialize(path,*args,**kwargs)

    def get_login_url(self) -> str:
        return self.application.auth_provider.login_url

    async def prepare(self):
        logger.debug("Preparing AuthedStatic etc")
        if self.require_auth:
            # For OAuth, it's async
            self.current_user = await self.application.auth_provider.get_user_async(self)
        else:
            # For BasicAuth (ie password), it's synchronous
            if self.application.auth_provider.get_user:
                self.current_user = self.application.auth_provider.get_user(self)
        return super().prepare()

    @authenticated
    async def get(self,path,*args,**kwargs):
        logger.debug(f"Authenticated and checking for access to {path}")
        authorized=(not self.require_auth)

        if self.require_auth:
            user_info=self.user_info
            if user_info:
                logger.info(f"Login attempt to {path} under '{self.root}'")
                if self.required_role in user_info.get('roles',[]):
                    logger.info(f"User has role '{self.required_role}', accepting login.")
                    authorized=True
                else:
                    logger.info(f"User does not have role '{self.required_role}', rejecting login.")
            else:
                logger.info(f"Non-login")
        else:
            logger.debug(f"Oauth disabled, providing {path}")

        if authorized:
            return await super().get(path,*args,**kwargs)
        else:
            self.path = self.parse_url_path(path)
            absolute_path = self.get_absolute_path(self.root, self.path)
            self.absolute_path = self.validate_absolute_path(self.root, absolute_path)
            self.set_status(403)

    @authenticated
    async def post(self,*args,**kwargs):
        raise NotImplementedError("Haven't implemented POST access for AuthedStaticFileHandler")

    def _decode_cookie(self, cookie_name, cookie=None):
        """ Modified from panel.io.state in Panel 1.3.6"""
        from tornado.web import decode_signed_value

        cookie = self.cookies.get(cookie_name)
        if cookie is None:
            return None
        # in the below line, the .value is different from panel
        cookie = decode_signed_value(config.cookie_secret, cookie_name, cookie.value)
        return self._decrypt_cookie(cookie)

    @property
    def encryption(self):
        return _pnstate.encryption

from panel.io.state import _state as _pnstate
AuthedStaticFileHandler.user_info=_pnstate.user_info
AuthedStaticFileHandler._decrypt_cookie=_pnstate._decrypt_cookie

### Copied this from panel.io.server, but modified to use my AuthedStaticFileHandler
def authed_get_static_routes(static_dirs):
    """
    Returns a list of tornado routes of StaticFileHandlers given a
    dictionary of slugs and file paths to serve.
    """
    patterns = []
    for slug, path in static_dirs.items():
        if type(path) is not str:
            path,role=path
        else:
            role=None
        if not slug.startswith('/'):
            slug = '/' + slug
        if slug == '/static':
            raise ValueError("Static file route may not use /static "
                             "this is reserved for internal use.")
        path = fullpath(path)
        if not os.path.isdir(path):
            raise ValueError("Cannot serve non-existent path %s" % path)
        need_auth=(get_auth_info()['oauth_provider']!='none')
        WhichStaticFileHandler=AuthedStaticFileHandler if need_auth else StaticFileHandler
        patterns.append(
            (r"%s/(.*)" % slug, WhichStaticFileHandler, {"path": path, "role": role} if need_auth else {"path": path})
        )
    patterns.append((
        f'/{COMPONENT_PATH}(.*)', ComponentResourceHandler, {}
    ))
    return patterns
def monkeypatch_authstaticroutes():
    panel.io.server.get_static_routes=authed_get_static_routes
    logger.debug("Extra static routes monkey-patched")

from tornado.web import RequestHandler
class SimpleSecretShare(RequestHandler):
    def initialize(self, callers:dict[str,Callable[[],str]]):
        self._callers=callers
    def get(self):
        self.write(f"Should be using POST")
    def post(self):
        accesskey=self.get_argument('access_key',default=None)
        if not accesskey:
            self.set_status(400)
            self.write("Need to supply access_key")
            return
        try:
            validated=AccessKeyDownload.validate_access_key(accesskey.encode())
        except BadSignatureError:
            self.set_status(403)
            self.write("Access key failed signature check!")
            return
        except: validated=False

        if not validated:
            self.set_status(403)
            self.write("Access key not valid")
            return
        age=datetime.now()-datetime.fromisoformat(validated['Generated'])
        if age>timedelta(days=90):
            self.set_status(403)
            self.write(f"Access key expired.  Its age is {age}.")
            return
        secretname=self.get_argument('secretname',default=None)
        if not secretname:
            self.set_status(400)
            self.write("Need to supply secretname")
            return
        if secretname not in self._callers:
            self.set_status(404)
            self.write(f"Secret '{secretname}' not recognized, options include {list(self._callers)}")
            return
        self.set_status(200)
        #self.write(f"You are {validated}.\n")
        self.write(self._callers[secretname]())

from datavac.appserve.app import PanelApp
import panel as pn
from io import StringIO
class AccessKeyDownload(PanelApp):
    def get_page(self):
        self.page.main.append(pn.Row(
            pn.layout.HSpacer(),
                pn.Column(
                    pn.pane.Markdown(
                        f"""
                        # Access Key
                        
                        This access key will allow DataVacuum code on your machine to retrieve temporary
                        database credentials to read our data.  Treat it like a password and do not share it.
                        """,width=500,dedent=True,renderer='markdown'),
                    pn.widgets.FileDownload(label='Download',icon='key',
                                          filename="datavacuum_access_key.txt",
                                          callback=self.generate_access_key,
                                          width=500),),
                pn.layout.HSpacer()))
        return self.page

    @staticmethod
    def generate_access_key() -> StringIO:
        user=pn.state.user_info['unique_name']
        gentime=str(datetime.now())
        bitstoverify=(gentime+user).encode()

        sign_seed=get_access_key_sign_seed()
        signing_key=nacl.signing.SigningKey(seed=sign_seed,encoder=Base64Encoder)

        sig=signing_key.sign(bitstoverify).signature
        key={'User':user,
             'Generated':gentime,
             'Signature':base64.b64encode(sig).decode()}

        sio=StringIO()
        yaml.dump(key,sio)
        sio.seek(0)

        logger.debug(f"Access key generated for {user}")
        sio.seek(0)

        return sio

    @staticmethod
    def validate_access_key(bytestring:bytes) -> dict:
        key=yaml.safe_load(bytestring.decode())
        user=key['User']
        gentime=key['Generated']
        sig=base64.b64decode(key['Signature'].encode())
        bitstoverify=(gentime+user).encode()

        sign_seed=get_access_key_sign_seed()
        verify_key=nacl.signing.SigningKey(seed=sign_seed,encoder=Base64Encoder).verify_key

        verify_key.verify(bitstoverify,sig)
        logger.debug("Signature accepted")
        return {k:v for k,v in key.items() if k!='Signature'}