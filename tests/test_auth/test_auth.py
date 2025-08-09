from contextlib import contextmanager
import time
from typing import Callable, cast
from datavac.util.dvlogging import logger


from tornado.web import RequestHandler
class MockOauthAuthorizeHandler(RequestHandler):
    user_logs_in_successfully = False
    def get(self):
        if self.user_logs_in_successfully:
            print("Got request for mock oauth authorize handler")
            print("redirecting to ...")
            print(f'{self.get_argument("redirect_uri")}?code=mock_code&state={self.get_argument("state")}')
            self.redirect(f'{self.get_argument("redirect_uri")}?code=mock_code&state={self.get_argument("state")}')
        else:
            self.write("User login failed")
        
class MockOauthTokenHandler(RequestHandler):
    user_info = {'unique_name': 'testuser', 'given_name':'Test User', 'roles': ['admin']}
    def post(self):
        print("Got POST request for mock oauth token handler")
        self.set_header("Content-Type", "application/json")
        import base64
        import json
        enc = base64.urlsafe_b64encode(json.dumps(self.user_info).encode('ascii')).decode('ascii')
        self.write({'access_token': enc, 'token_type': 'bearer'})

from panel.auth import AUTH_PROVIDERS, AzureAdLoginHandler
class MockAzureAdLoginHandler(AzureAdLoginHandler):
    @property
    def _OAUTH_ACCESS_TOKEN_URL_(self):
        return f'http://localhost:{PCONF().server_config.port}/mock_oauth2_token'
    @property
    def _OAUTH_AUTHORIZE_URL_(self):
        return f'http://localhost:{PCONF().server_config.port}/mock_oauth2_authorize'
AUTH_PROVIDERS['mock_azure']=MockAzureAdLoginHandler

from datavac.appserve.ad_auth import UsePanelAuthMixin
class SensistiveDataHandler(UsePanelAuthMixin,RequestHandler):
    required_role = 'admin'  # type: ignore
    require_token_method = True
    yes_message="Access granted to sensitive data."
    no_message="Not authorized."
    async def get_authorized(self, *args, **kwargs):
        self.write(self.yes_message)
    async def get_not_authorized(self, *args, **kwargs):
        self.set_status(403)
        self.write(self.no_message)


from datavac.config.project_config import ProjectConfiguration, PCONF
from datavac.config.server_config import ServerConfig
from datavac.appserve.dvsecrets.vaults.vault import Vault
from pathlib import Path
class TestServerConfig(ServerConfig):
    def get_yaml(self) -> dict:
        return {'index':{'':{'':{'app':'datavac.appserve.index:AppIndex','role':'admin'}}},
                'additional_static_dirs':{'teststatic':{'path':Path(__file__).parent,'role':'admin'}}}
    def get_additional_handlers(self):
        return {'mock_oauth2_token': MockOauthTokenHandler,
                'mock_oauth2_authorize': MockOauthAuthorizeHandler,
                'sensitive': SensistiveDataHandler,}
class MockVault(Vault):
    def get_auth_info(self):
        #return {'oauth_provider':'none'}
        return {
            'oauth_provider':'mock_azure',
            'oauth_key':'MyApplicationID',
            'oauth_extra_params':{'tenant_id':'MyTenantID'},
            'oauth_secret':'MySecret',
            'oauth_redirect_uri':f'http://localhost:{PCONF().server_config.port}/',
            'cookie_secret':'MyCookieSecret',}
    def get_access_key_sign_seed(self) -> bytes:
        return b'TestSeed'
PCONF(ProjectConfiguration(deployment_name='datavac_authtest',
    data_definition=None,  # type: ignore
    vault=MockVault(),
    cert_depo=None,  # type: ignore
    server_config=TestServerConfig()
))

#def test_something():
#    import panel as pn
#    from datavac.appserve.panel_serve import launch
#    server=launch()#non_blocking=True)
#    #import requests
#    #print(requests.get(f'http://localhost:{PCONF().server_config.port}/mock_oauth',timeout=.5).text)
#    #time.sleep(10)
#    #server.stop()

@contextmanager
def mock_webbrowser_with_playwright():
    import playwright
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        class MockWebBrowser:
            def open(self, url, new=1):
                print(f"Mock web browser opening URL: {url}")
                page.goto(url)
        yield MockWebBrowser()
        browser.close()

def test_ak_download():
    from datavac.appserve.panel_serve import launch
    server=launch(non_blocking=True)
    from datavac.appserve.dvsecrets.ak_client_side import direct_user_to_access_key, get_access_key, invalidate_access_key, _access_key_path
    prev_user_logs_in_successfully = MockOauthAuthorizeHandler.user_logs_in_successfully
    try:
        import jwt, webbrowser
        should_be_jwt = jwt.encode(MockOauthTokenHandler.user_info, 'TestSeed', algorithm='HS256')

        MockOauthAuthorizeHandler.user_logs_in_successfully = True
        invalidate_access_key()  # Ensure no cached access key
        assert not _access_key_path().exists()
        with mock_webbrowser_with_playwright() as mock_browser:
            received_jwt=get_access_key(specific_webbrowser=cast(webbrowser.BaseBrowser,mock_browser), login_timeout=2)
        assert received_jwt == should_be_jwt
        assert _access_key_path().exists()
        logger.info("Successfully received API key when user can log in")

        import pytest
        from datavac.appserve.dvsecrets.ak_client_side import AccessKeyRetrievalException
        invalidate_access_key()  # Ensure no cached access key
        assert not _access_key_path().exists()
        with pytest.raises(AccessKeyRetrievalException):
            MockOauthAuthorizeHandler.user_logs_in_successfully = False
            with mock_webbrowser_with_playwright() as mock_browser:
                received_jwt=get_access_key(specific_webbrowser=cast(webbrowser.BaseBrowser,mock_browser), login_timeout=2)
        assert not _access_key_path().exists()
        logger.info("Successfully *did not* receive API key when user *cannot* log in")
    finally:
        MockOauthAuthorizeHandler.user_logs_in_successfully = prev_user_logs_in_successfully
        server.stop()

def test_ak_access():
    from datavac.appserve.panel_serve import launch
    server=launch(non_blocking=True)
    time.sleep(1)
    import requests
    import jwt
    try: 
        # Try with everything correct
        jwt_token = jwt.encode({'unique_name': 'testuser', 'given_name': 'Test User', 'roles': ['admin']}, 'TestSeed', algorithm='HS256')
        resp = requests.get(f'http://localhost:{PCONF().server_config.port}/sensitive',timeout=.5,allow_redirects=False,
                           headers={'Authorization':f'Bearer {jwt_token}'})
        print(resp.text)
        assert resp.text == SensistiveDataHandler.yes_message
        
        # Try with wrong seed
        jwt_token = jwt.encode({'unique_name': 'testuser', 'given_name': 'Test User', 'roles': ['admin']}, 'BadTestSeed', algorithm='HS256')
        resp = requests.get(f'http://localhost:{PCONF().server_config.port}/sensitive',timeout=.5,allow_redirects=False,
                           headers={'Authorization':f'Bearer {jwt_token}'})
        print(resp.text)
        assert "Failed" in resp.text
        assert resp.status_code == 400
        assert SensistiveDataHandler.yes_message not in resp.text

        # Try with wrong role
        jwt_token = jwt.encode({'unique_name': 'testuser', 'given_name': 'Test User', 'roles': ['not-admin']}, 'TestSeed', algorithm='HS256')
        resp = requests.get(f'http://localhost:{PCONF().server_config.port}/sensitive',timeout=.5,allow_redirects=False,
                           headers={'Authorization':f'Bearer {jwt_token}'}).text
        print(resp)
        assert resp == SensistiveDataHandler.no_message
    finally:
        server.stop()
     

if __name__ == '__main__':
    #test_something()
    test_ak_download()
    test_ak_access()