from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional
from datavac.appserve.auth import UsePanelAuthMixin
from datavac.util.dvlogging import logger
from nacl.exceptions import BadSignatureError
class APIKeyGenerator(UsePanelAuthMixin):
    required_role = None
    require_browser_method = True
    async def get_authorized(self, user_info: dict[str, Any]):
        """ Generates an API key for the user. """
        print("Got request to generate API key")
        import jwt
        from datavac.config.project_config import PCONF
        sign_seed=PCONF().vault.get_access_key_sign_seed()
        user_info=user_info.copy()
        logger.debug("Sharing JWT for: ")
        logger.debug(str(user_info))
        raw_jwt=jwt.encode(user_info,sign_seed,algorithm='HS256',)

        
        port=self.get_argument('redirect_port',default=None)
        if (port is None) or (not port.isdigit()) or (int(port)<=0) or (int(port)>65535):
            self.set_status(403)
            self.write("Invalid or absent redirect_port, cannot generate API key URL")
            return
        b64_pkey=self.get_argument('pkey',default=None)
        if b64_pkey is None:
            self.set_status(403)
            self.write("No pkey provided, cannot securely share API key")
            return
        logoutfirst=self.get_argument('logoutfirst',default='false').lower() in ['true','1','t','yes','y']
        if logoutfirst:
            self.set_header("Clear-Site-Data", '"cache","cookies","storage"')
            self.write("Your cookies should be clear!")
            self.redirect(f'/accesskey?pkey={b64_pkey}&redirect_port={port}')
            return

        import base64
        from nacl.public import SealedBox, PublicKey
        pkey=PublicKey(base64.urlsafe_b64decode(b64_pkey.encode('utf-8')))
        message=base64.urlsafe_b64encode(SealedBox(pkey).encrypt(raw_jwt.encode())).decode('utf-8')
        self.redirect(f'http://localhost:{port}/receive_api_key?message={message}')


#def validate_access_key(bytestring:bytes, required_role:Optional[str] = None) -> dict:
#    import jwt
#    from datavac.config.project_config import PCONF
#    key=jwt.decode(bytestring.decode(),PCONF().vault.get_access_key_sign_seed(),algorithms=['HS256'])
#    user=key['User']
#    logger.debug("Signature accepted")
#
#    assert 'Generated' in key
#    age=datetime.now()-datetime.fromisoformat(key['Generated'])
#    if age>timedelta(days=PCONF().server_config.access_key_expiry_days):
#        raise AccessKeyExpiredError(f"Access key expired.  Its age is {age}, max allowed is {PCONF().server_config.access_key_expiry_days} days.")
#
#    if required_role is not None:
#        if required_role not in key.get('roles',[]):
#            logger.warning(f"Access key does not have required role '{required_role}'")
#            raise PermissionError(f"Access key does not have required role '{required_role}'")
#        else: logger.debug(f"User {user} has '{required_role}'")
#    else: logger.debug(f"Access key does not require role check")
#
#    return {k:v for k,v in key.items() if k!='Signature'}


#def with_role(required_role: Optional[str]=None) -> Callable:
#    """ Decorator to check if the user has the required role before calling the function."""
#    def decorator(func: Callable) -> Callable:
#        @wraps(func)
#        def wrapper(self, *args, **kwargs):
#            """ Wrapper function that validates the access key before calling the original function.
#
#            Args:
#                self: The instance of the class where the function is defined.
#                *args: Positional arguments for the original function.
#                **kwargs: Keyword arguments for the original function.
#            """
#            accesskey=self.get_argument('access_key',default=None)
#            if not accesskey:
#                self.set_status(400)
#                self.write("Need to supply access_key")
#                return
#            try:
#                validated=validate_access_key(accesskey.encode(),required_role=required_role)
#            except BadSignatureError:
#                self.set_status(403)
#                self.write("Access key failed signature check!")
#                return
#            except PermissionError as e:
#                self.set_status(403)
#                self.write(str(e))
#                return
#            except: validated=False
#
#            if not validated:
#                self.set_status(403)
#                self.write("Access key not valid")
#                return
#            age=datetime.now()-datetime.fromisoformat(validated['Generated'])
#            if age>timedelta(days=90):
#                self.set_status(403)
#                self.write(f"Access key expired.  Its age is {age}.")
#                return
#
#            # Call the original function after validation
#            return func(self, *args, validated=validated, **kwargs)
#        return wrapper
#    return decorator
#
#
#def with_validated_access_key(func:Callable) -> Callable:
#    return with_role(required_role=None)(func)

