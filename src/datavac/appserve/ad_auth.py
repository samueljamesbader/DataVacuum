import base64
from functools import wraps
import shutil

from typing import Optional, cast
import yaml
import os
from datetime import datetime, timedelta
from typing import Callable

import nacl.signing
import panel
from nacl.encoding import Base64Encoder
from nacl.exceptions import BadSignatureError
from panel.auth import AzureAdLoginHandler, OAuthProvider

from datavac.util.dvlogging import logger

### This part should be unnecessary now that in 1.7.5, Panel supports authentication for static files
from panel.util import fullpath, decode_token
from panel.io.resources import COMPONENT_PATH
from panel.io.server import ComponentResourceHandler
from tornado.web import StaticFileHandler, authenticated
### from panel.config import config
#class UsePanelAuthMixin:
#    """ Mixin to use Panel's authentication system in Tornado RequestHandlers.
#
#    Important: make sure to place this mixin before any RequestHandler in the inheritance list.
#    ```python
#    # This is good
#    class MyHandler(UsePanelAuthMixin, RequestHandler): pass
#
#    # This is bad
#    # The UsePanelAuthMixin.get() method will be overridden by RequestHandler.get()
#    class MyHandler(RequestHandler, UsePanelAuthMixin): pass
#    '''
#    
#    """
#    @property
#    def require_auth(self) -> bool:
#        from datavac.config.project_config import PCONF
#        return (PCONF().vault.get_auth_info()['oauth_provider']!="none")
#    def get_required_role(self) -> Optional[str]:
#        if hasattr(self, 'required_role'):
#            return self.required_role # type: ignore
#        else:
#            raise NotImplementedError("Must implement get_required_role method in subclass or set self.required_role")
#    def get_login_url(self) -> str:
#        print(f"giving login url as {self.application.auth_provider.login_url}")
#        return self.application.auth_provider.login_url
#    async def prepare(self):
#        logger.debug("Preparing AuthedStatic etc")
#        if self.require_auth:
#            # For OAuth, it's async
#            self.current_user = await self.application.auth_provider.get_user_async(self)
#        else:
#            # For BasicAuth (ie password), it's synchronous
#            if self.application.auth_provider.get_user:
#                self.current_user = self.application.auth_provider.get_user(self)
#        print(f"Prepared {self.current_user}")
#        return super().prepare()
#    def _decode_cookie(self, cookie_name, cookie=None)->str|None:
#        """ Modified from panel.io.state in Panel 1.3.6"""
#        from tornado.web import decode_signed_value
#        from panel.config import config
#
#        cookie = self.cookies.get(cookie_name)
#        if cookie is None: return None
#        # in the below line, the .value is different from panel
#        cookie = decode_signed_value(config.cookie_secret, cookie_name, cookie.value) # type: ignore
#        return pn.state._decrypt_cookie(cookie)
#    
#    @authenticated
#    async def get(self,*args,**kwargs):
#        path=self.request.path
#        logger.debug(f"Authenticated and checking for access to {path}")
#        authorized=(not self.require_auth)
#
#        if self.require_auth:
#
#            if not self.require_token_method:            
#                user_info_token=self._decode_cookie('id_token')
#                if user_info_token is None:
#                    logger.info("Bad id_token")
#                else:
#                    user_info=decode_token(user_info_token)
#            if user_info is None:
#                auth_header=self.get_header('Authorization')
#                print(auth_header)
#                assert False
#                
#            print(user_info)
#            if user_info:
#                logger.info(f"Login attempt to {path}")
#                required_role=self.get_required_role()
#                if required_role in user_info.get('roles',[]):
#                    logger.info(f"User has role '{required_role}', accepting login.")
#                    authorized=True
#                else:
#                    logger.info(f"User does not have role '{required_role}', rejecting login.")
#            else:
#                logger.info(f"Non-login")
#        else:
#            logger.debug(f"Oauth disabled, providing {path}")
#
#        if authorized:
#            await self.get_authorized(path,*args,**kwargs)
#        else:
#            await self.get_not_authorized(path,*args,**kwargs)
#
#    async def get_not_authorized(self, path, *args, **kwargs):
#        raise NotImplementedError("Must implement get_not_authorized method in subclass")
#    async def get_authorized(self, path, *args, **kwargs):
#        raise NotImplementedError("Must implement get_authorized method in subclass")

class UsePanelAuthMixin:
    """ Mixin to use Panel's authentication system in Tornado RequestHandlers.

    Important: make sure to place this mixin before any RequestHandler in the inheritance list.
    ```python
    # This is good
    class MyHandler(UsePanelAuthMixin, RequestHandler): pass

    # This is bad
    # The UsePanelAuthMixin.get() method will be overridden by RequestHandler.get()
    class MyHandler(RequestHandler, UsePanelAuthMixin): pass
    '''
    """
    require_token_method: bool = False
    async def get(self, *args, **kwargs):

        assert isinstance(self, RequestHandler)
        auth_header=self.request.headers.get('Authorization',default=None)
        if self.require_token_method and auth_header is None:
            self.set_status(400)
            self.write("No Authorization header provided, cannot authenticate")
            return
        if auth_header is not None:
            if not auth_header.startswith('Bearer '):
                self.set_status(400)
                self.write("Authorization header must start with 'Bearer '")
                return
            import jwt
            from datavac.config.project_config import PCONF
            sign_seed=PCONF().vault.get_access_key_sign_seed()
            try:
                user_info=jwt.decode(auth_header[7:],sign_seed,algorithms=['HS256'])
            except Exception as e:
                self.set_status(400)
                self.write(f"Failed to decode token from Authorization header: {str(e)}")
                return
            get_from='header'
        else:
            auth_provider=self.application.auth_provider # type: ignore
            if hasattr(auth_provider, 'get_user_async'):
                self.current_user = await auth_provider.get_user_async(self)
            else: self.current_user = auth_provider.get_user(self)
            get_from='cookie'
        
        async def get_with_auth(self, *args, **kwargs):
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
                    return await self.get_authorized(*args, user_info=user_info, **kwargs)
                else:
                    if required_role in user_info.get('roles',[]):
                        logger.info(f"User has role '{required_role}', accepting login.")
                        return await self.get_authorized(*args, user_info=user_info, **kwargs)
                    else:
                        logger.info(f"User does not have role '{required_role}', rejecting login.")
            else:
                logger.info(f"Non-login")
            return await self.get_not_authorized(*args, user_info=user_info, **kwargs)

        if get_from=='cookie':
            from tornado.web import authenticated
            ares=authenticated(get_with_auth)(self, *args, **kwargs)
            if ares is not None: await ares
        else:
            await get_with_auth(self, *args, **kwargs)

    def get_login_url(self) -> str:
        assert isinstance(self, RequestHandler)
        auth_provider = self.application.auth_provider # type: ignore
        print(f"giving login url as {auth_provider.login_url}")
        return auth_provider.login_url
        
    def _decode_cookie(self, cookie_name, cookie=None)->str|None:
        """ Modified from panel.io.state in Panel 1.3.6"""
        from tornado.web import decode_signed_value
        from panel.config import config

        assert isinstance(self, RequestHandler)
        cookie = self.cookies.get(cookie_name)
        if cookie is None: return None
        # in the below line, the .value is different from panel
        cookie = decode_signed_value(config.cookie_secret, cookie_name, cookie.value) # type: ignore
        return pn.state._decrypt_cookie(cookie)
    
    async def get_not_authorized(self, *args, **kwargs):
        assert isinstance(self, RequestHandler)
        self.set_status(403)
        
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

def with_validated_access_key(func:Callable) -> Callable:
    return with_role(required_role=None)(func)

def with_role(required_role: Optional[str]=None) -> Callable:
    """ Decorator to check if the user has the required role before calling the function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """ Wrapper function that validates the access key before calling the original function.
            
            Args:
                self: The instance of the class where the function is defined.
                *args: Positional arguments for the original function.
                **kwargs: Keyword arguments for the original function.
            """
            accesskey=self.get_argument('access_key',default=None)
            if not accesskey:
                self.set_status(400)
                self.write("Need to supply access_key")
                return
            try:
                validated=AccessKeyDownload.validate_access_key(accesskey.encode(),required_role=required_role)
            except BadSignatureError:
                self.set_status(403)
                self.write("Access key failed signature check!")
                return
            except PermissionError as e:
                self.set_status(403)
                self.write(str(e))
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
            
            # Call the original function after validation
            return func(self, *args, validated=validated, **kwargs)
        return wrapper
    return decorator

from tornado.web import RequestHandler
class SimpleSecretShare(RequestHandler):
    def get(self):
        self.write(f"Should be using POST")

    def initialize(self, callers:dict[str,Callable[[],str]]):
        self._callers=callers

    @with_validated_access_key
    def post(self, validated:dict[str,str]):
        """ Handles POST requests to retrieve secrets."""
        
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
        self.write(self._callers[secretname]())


from datavac.appserve.app import PanelApp
import panel as pn
from io import StringIO
class AccessKeyDownload(PanelApp):
    def get_page(self):
        self.page.main.append(pn.Row( # type: ignore
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
                                          callback=lambda: StringIO(self.generate_access_key()),
                                          width=500),),
                pn.layout.HSpacer()))
        return self.page

    @staticmethod
    def generate_access_key() -> str:
        import jwt
        from datavac.config.project_config import PCONF
        user_info=pn.state.user_info
        assert user_info is not None, "User info must be set in Panel state"
        user=['unique_name']
        gentime=str(datetime.now())

        sign_seed=PCONF().vault.get_access_key_sign_seed()
        return jwt.encode({'User':user,'Generated':gentime, 'roles':user_info.get('roles',[])},sign_seed,algorithm='HS256',)

    @staticmethod
    def validate_access_key(bytestring:bytes, required_role:Optional[str] = None) -> dict:
        import jwt
        from datavac.config.project_config import PCONF
        key=jwt.decode(bytestring.decode(),PCONF().vault.get_access_key_sign_seed(),algorithms=['HS256'])
        user=key['User']
        logger.debug("Signature accepted")
        
        if required_role is not None:
            if required_role not in key.get('roles',[]):
                logger.warning(f"Access key does not have required role '{required_role}'")
                raise PermissionError(f"Access key does not have required role '{required_role}'")
            else: logger.debug(f"User {user} has '{required_role}'")
        else: logger.debug(f"Access key does not require role check")

        return {k:v for k,v in key.items() if k!='Signature'}