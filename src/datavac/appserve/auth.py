
import time
from datetime import datetime, timedelta
from typing import Optional, cast
import os
from typing import Callable

import panel

from datavac.util.dvlogging import logger

from panel.util import fullpath, decode_token
from panel.io.resources import COMPONENT_PATH
from panel.io.server import ComponentResourceHandler
from tornado.web import StaticFileHandler, RequestHandler

class UsePanelAuthMixin(RequestHandler):
    """ Mixin to use Panel's authentication system or a DV access key in Tornado RequestHandlers.

    Instead of defining a get() method, define get_authorized() and get_not_authorized() methods.
    If there is an authentication failure that prevents getting a user_info dict, will respond with a 400
    Otherwise, calls get_authorized() or get_not_authorized() depending whether the user has the required role.
    Both functions will receive an additional keyword argument 'user_info', which is a dict with user information,
    in addition to the *args and **kwargs passed to get().  Same advice for post().

    Important: make sure to place this mixin before any RequestHandler descendant in the inheritance list.
    ```python
    # This is good
    class MyHandler(UsePanelAuthMixin, DescendantOfRequestHandler): pass

    # This is bad
    # The UsePanelAuthMixin.get() method will be overridden by RequestHandler.get()
    class MyHandler(DescendantOfRequestHandler, UsePanelAuthMixin): pass # bad
    '''
    """
    require_token_method: bool = False
    require_browser_method: bool = False
    required_role: Optional[str] = ... # type: ignore
    async def called(self, called_authorized,called_not_authorized, *args, **kwargs):

        auth_header=self.request.headers.get('Authorization',default=None)
        if self.require_token_method and (auth_header is None):
            self.set_status(401)
            self.write("No Authorization header provided, cannot authenticate")
            return
        if (not self.require_browser_method) and (auth_header is not None):
            if not auth_header.startswith('Bearer '):
                self.set_status(401)
                self.write("Authorization header must start with 'Bearer '")
                return
            import jwt
            from datavac.config.project_config import PCONF
            sign_seed=PCONF().vault.get_access_key_sign_seed()
            aud=PCONF().vault.get_auth_info().get('oauth_key',None)
            try:
                user_info=jwt.decode(auth_header[7:],sign_seed,algorithms=['HS256'],audience=aud,
                                     options={'verify_exp': False,'verify_aud':(aud is not None)}) # we check expiry ourselves
            except Exception as e:
                self.set_status(401)
                self.write(f"Failed to decode token from Authorization header: {str(e)}")
                return
            try:
                age=time.time()-user_info['iat']
                if timedelta(seconds=age)>timedelta(days=14): # TODO: make configurable
                    self.set_status(401)
                    self.write(f"Access key expired.  Its age is {age}.")
                    return
            except Exception as e:
                self.set_status(401)
                self.write(f"Could not determine age of access key")
                # print traceback
                import traceback
                print(traceback.format_exc()) # TODO: remove
                return
            
            get_from='header'
        else:
            auth_provider=self.application.auth_provider # type: ignore
            if hasattr(auth_provider, 'get_user_async'):
                self.current_user = await auth_provider.get_user_async(self)
            else: self.current_user = auth_provider.get_user(self)
            get_from='cookie'
        
        async def call_with_auth(self:'UsePanelAuthMixin', *args, **kwargs):
            nonlocal user_info
            if get_from=='cookie':
                user_info_token=self._decode_cookie('id_token')
                if user_info_token is None:
                    logger.info("Bad id_token")
                    user_info=None
                else:
                    user_info=decode_token(user_info_token)
                    if type(user_info) is not dict:
                        logger.info("Failed at decoding token")
                        user_info=None
            print(user_info)
            if user_info:
                path= self.request.path
                logger.info(f"Login attempt to {path}")
                required_role=self.required_role
                if required_role is None:
                    logger.info(f"Role not required for '{path}', accepting login.")
                    return await called_authorized(*args, user_info=user_info, **kwargs)
                else:
                    if required_role in user_info.get('roles',[]):
                        logger.info(f"User has role '{required_role}', accepting login.")
                        return await called_authorized(*args, user_info=user_info, **kwargs)
                    else:
                        logger.info(f"User does not have role '{required_role}', rejecting login.")
            else:
                logger.info(f"Non-login")
            return await called_not_authorized(*args, user_info=user_info, **kwargs)

        if get_from=='cookie':
            from tornado.web import authenticated
            ares=authenticated(call_with_auth)(self, *args, **kwargs)
            if ares is not None: await ares
        else:
            await call_with_auth(self, *args, **kwargs)

    async def get(self, *args, **kwargs):
        if hasattr(self, 'get_authorized'):
            await self.called(self.get_authorized,self.get_not_authorized, *args, **kwargs) # type: ignore
        else:
            self.set_status(405)
            self.write("GET method not allowed")
    async def post(self, *args, **kwargs):
        if hasattr(self, 'post_authorized'):
            await self.called(self.post_authorized,self.post_not_authorized, *args, **kwargs) # type: ignore
        else:
            self.set_status(405)
            self.write("POST method not allowed")

    def get_login_url(self) -> str:
        assert isinstance(self, RequestHandler)
        auth_provider = self.application.auth_provider # type: ignore
        print(f"giving login url as {auth_provider.login_url}")
        return auth_provider.login_url
        
    def _decode_cookie(self, cookie_name, cookie=None)->str|None:
        """ Modified from panel.io.state in Panel 1.3.6"""
        from tornado.web import decode_signed_value
        from panel.config import config
        import panel as pn

        assert isinstance(self, RequestHandler)
        cookie = self.cookies.get(cookie_name)
        if cookie is None: return None
        # in the below line, the .value is different from panel
        cookie = decode_signed_value(config.cookie_secret, cookie_name, cookie.value) # type: ignore
        return pn.state._decrypt_cookie(cookie)
    
    async def get_not_authorized(self, *args, **kwargs):
        assert isinstance(self, RequestHandler)
        self.set_status(403)
        self.write(f"You don't have permission to access this resource. Required role: {self.required_role}")
    async def post_not_authorized(self, *args, **kwargs):
        assert isinstance(self, RequestHandler)
        self.set_status(403)
        self.write(f"You don't have permission to access this resource. Required role: {self.required_role}")
    def initialize(self, *args, **kwargs):
        try:
            assert self.get._comes_from_usepanelauthmixin # type: ignore
            assert self.post._comes_from_usepanelauthmixin # type: ignore
        except Exception:
            raise RuntimeError("UsePanelAuthMixin must be placed before RequestHandler in the inheritance list")
        super().initialize(*args, **kwargs)
UsePanelAuthMixin.get._comes_from_usepanelauthmixin=True # type: ignore
UsePanelAuthMixin.post._comes_from_usepanelauthmixin=True # type: ignore


class AuthedStaticFileHandler(UsePanelAuthMixin,StaticFileHandler):
    require_token_method: bool = False
    def initialize(self, path: str, role: str|None):
        self.required_role=role
        return super().initialize(path)
    async def get_not_authorized(self, path, *args, user_info=None, **kwargs):
        self.path = self.parse_url_path(path)
        absolute_path = self.get_absolute_path(self.root, self.path)
        self.absolute_path = self.validate_absolute_path(self.root, absolute_path)
        self.set_status(403)
    async def get_authorized(self, path, *args, user_info=None, **kwargs):
        print(path, *args)
        return await StaticFileHandler.get(self, path ,*args,**kwargs)

#def handler_with_auth(func: Callable, required_role: str|None):
#    class Handler(UsePanelAuthMixin, RequestHandler):
#        require_token_method: bool = True
#        def __init__(self,*args,**kwargs) -> None:
#            self.required_role=required_role
#            super().__init__(*args,**kwargs)
#        def get_authorized(self, *args, user_info=None, **kwargs):
#            return func(self, *args, user_info=user_info, **kwargs)
#    return Handler




### from panel.io.state import _state as _pnstate
### AuthedStaticFileHandler.user_info=_pnstate.user_info
### AuthedStaticFileHandler._decrypt_cookie=_pnstate._decrypt_cookie
### 
### Copied this from panel.io.server, but modified to use my AuthedStaticFileHandler
def authed_get_static_routes(static_dirs):
    """
    Returns a list of tornado routes of StaticFileHandlers given a
    dictionary of slugs and file paths to serve.
    """
    from datavac.config.project_config import PCONF
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
        need_auth=(PCONF().vault.get_auth_info()['oauth_provider']!='none')
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

#from tornado.web import RequestHandler
# from datavac.appserve.dvsecrets.ak_server_side import with_validated_access_key
#class SimpleSecretShare(RequestHandler):
#    def get(self):
#        self.write(f"Should be using POST")
#
#    def initialize(self, callers:dict[str,Callable[[],str]]):
#        self._callers=callers
#
#    @with_validated_access_key
#    def post(self, validated:dict[str,str]):
#        """ Handles POST requests to retrieve secrets."""
#        
#        secretname=self.get_argument('secretname',default=None)
#        if not secretname:
#            self.set_status(400)
#            self.write("Need to supply secretname")
#            return
#        if secretname not in self._callers:
#            self.set_status(404)
#            self.write(f"Secret '{secretname}' not recognized, options include {list(self._callers)}")
#            return
#        self.set_status(200)
#        self.write(self._callers[secretname]())

