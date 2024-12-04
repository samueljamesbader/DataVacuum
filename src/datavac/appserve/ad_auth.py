import os

import panel
from panel.auth import AzureAdLoginHandler, OAuthProvider
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
        self.require_auth=(os.environ['DATAVACUUM_OAUTH_PROVIDER']!="none")
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
        patterns.append(
            (r"%s/(.*)" % slug, AuthedStaticFileHandler, {"path": path, "role": role})
        )
    patterns.append((
        f'/{COMPONENT_PATH}(.*)', ComponentResourceHandler, {}
    ))
    return patterns
def monkeypatch_authstaticroutes():
    panel.io.server.get_static_routes=authed_get_static_routes
    logger.debug("Extra static routes monkey-patched to support authentication")
