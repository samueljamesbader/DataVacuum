from datetime import datetime
from typing import Any
from datavac.appserve.ad_auth import UsePanelAuthMixin
import panel as pn
from tornado.httputil import HTTPServerRequest
from tornado.web import Application, RequestHandler
from tornado.web import StaticFileHandler, authenticated
class APIKeyGenerator(UsePanelAuthMixin,RequestHandler):
    required_role = None
    async def get_authorized(self, user_info: dict[str, Any]):
        """ Generates an API key for the user. """
        print("Got request to generate API key")
        import jwt
        from datavac.config.project_config import PCONF
        sign_seed=PCONF().vault.get_access_key_sign_seed()
        raw_jwt=jwt.encode(user_info,sign_seed,algorithm='HS256',)

        b64_pkey=self.get_argument('pkey',default=None)
        if b64_pkey is None:
            self.set_status(403)
            self.write("No pkey provided, cannot securely share API key")
        else:
            import base64
            from nacl.public import SealedBox, PublicKey
            pkey=PublicKey(base64.urlsafe_b64decode(b64_pkey.encode('utf-8')))
            message=base64.urlsafe_b64encode(SealedBox(pkey).encrypt(raw_jwt.encode())).decode('utf-8')
            port=self.get_argument('redirect_port',default=None)
            if port is None:
                self.set_status(403)
                self.write("No redirect_port provided, cannot generate API key URL")
            else:
                self.redirect(f'http://localhost:{port}/receive_api_key?message={message}')
            
