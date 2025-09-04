from contextlib import contextmanager
import time
from typing import Callable, cast
import os

import pytest

@pytest.fixture(scope='module',autouse=True)
def mock_env_demo2_auth():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DATAVACUUM_CONTEXT", "builtin:demo2")
        from datavac import unload_my_imports; unload_my_imports()
        setup_auth_test()
        print("set up mock environment for demo2")
        yield
        from datavac import unload_my_imports; unload_my_imports()

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
        import datetime
        from datavac.config.project_config import PCONF
        aud=PCONF().vault.get_auth_info().get('oauth_key',None)
        user_info=self.user_info.copy()
        user_info['iat']= int(time.time())
        user_info['aud']= aud
        enc = base64.urlsafe_b64encode(json.dumps(user_info).encode('ascii')).decode('ascii')
        self.write({'access_token': enc, 'token_type': 'bearer'})

from panel.auth import AUTH_PROVIDERS, AzureAdLoginHandler
class MockAzureAdLoginHandler(AzureAdLoginHandler):
    @property
    def _OAUTH_ACCESS_TOKEN_URL_(self):
        from datavac.config.project_config import PCONF
        return f'http://localhost:{PCONF().server_config.port}/mock_oauth2_token'
    @property
    def _OAUTH_AUTHORIZE_URL_(self):
        from datavac.config.project_config import PCONF
        return f'http://localhost:{PCONF().server_config.port}/mock_oauth2_authorize'
AUTH_PROVIDERS['mock_azure']=MockAzureAdLoginHandler



sensitive_yes_message="Access granted to sensitive data."
sensitive_no_message="Not authorized."
def setup_auth_test():
    from datavac import unload_my_imports; unload_my_imports()
    from datavac.appserve.auth import UsePanelAuthMixin
    class SensistiveDataHandler(UsePanelAuthMixin):
        required_role = 'admin'  # type: ignore
        require_token_method = True
        async def get_authorized(self, *args, **kwargs):
            self.write(sensitive_yes_message)
        async def get_not_authorized(self, *args, **kwargs):
            self.set_status(403)
            self.write(sensitive_no_message)


    from datavac.config.server_config import ServerConfig
    from datavac.appserve.dvsecrets.vaults.vault import Vault
    from datavac.config.cert_depo import CertDepo
    from datavac.util.dvlogging import logger
    from pathlib import Path
    class TestServerConfig(ServerConfig):
        def get_yaml(self) -> dict:
            return {'index':{'':{'':{'app':'datavac.appserve.index:AppIndex','role':'admin'}}},
                    'additional_static_dirs':{'teststatic':{'path':Path(__file__).parent,'role':'admin'}}}
        def get_additional_handlers(self):
            return {'/mock_oauth2_token': MockOauthTokenHandler,
                    '/mock_oauth2_authorize': MockOauthAuthorizeHandler,
                    '/sensitive': SensistiveDataHandler,}
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
    from datavac.config.project_config import ProjectConfiguration, PCONF
    PCONF(ProjectConfiguration(deployment_name='datavac_authtest',
        data_definition=None,  # type: ignore
        vault=MockVault(),
        cert_depo=CertDepo(),
        server_config=TestServerConfig()
    ))
    logger.debug("Set up auth test configuration")

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
    from datavac.config.project_config import PCONF
    from datavac.appserve.panel_serve import launch
    from datavac.util.dvlogging import logger
    server=launch(non_blocking=True)
    from datavac.appserve.dvsecrets.ak_client_side import direct_user_to_access_key, get_access_key, invalidate_access_key, _access_key_path
    prev_user_logs_in_successfully = MockOauthAuthorizeHandler.user_logs_in_successfully
    aud=PCONF().vault.get_auth_info().get('oauth_key',None)
    try:
        import jwt, webbrowser
        #should_be_jwt = jwt.encode(MockOauthTokenHandler.user_info, PCONF().vault.get_access_key_sign_seed(), algorithm='HS256')

        MockOauthAuthorizeHandler.user_logs_in_successfully = True
        invalidate_access_key()  # Ensure no cached access key
        assert not _access_key_path().exists()
        with mock_webbrowser_with_playwright() as mock_browser:
            received_jwt=get_access_key(specific_webbrowser=cast(webbrowser.BaseBrowser,mock_browser), login_timeout=2)
        assert {k:v for k,v in jwt.decode(received_jwt, PCONF().vault.get_access_key_sign_seed(), audience=aud,algorithms=['HS256']).items()
                if k not in ['iat','aud']} == MockOauthTokenHandler.user_info
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
    from datavac.config.project_config import PCONF
    from datavac.appserve.panel_serve import launch
    server=launch(non_blocking=True)
    time.sleep(1)
    import requests
    import jwt
    goodseed=PCONF().vault.get_access_key_sign_seed()
    aud=PCONF().vault.get_auth_info().get('oauth_key',None)
    try: 
        # Try with everything correct
        jwt_token = jwt.encode({'unique_name': 'testuser', 'given_name': 'Test User', 'roles': ['admin'], 'aud': aud, 'iat':time.time()}, goodseed, algorithm='HS256')
        resp = requests.get(f'http://localhost:{PCONF().server_config.port}/sensitive',timeout=.5,allow_redirects=False,
                           headers={'Authorization':f'Bearer {jwt_token}'})
        print(resp.text)
        assert resp.text == sensitive_yes_message
        
        # Try with wrong seed
        jwt_token = jwt.encode({'unique_name': 'testuser', 'given_name': 'Test User', 'roles': ['admin'], 'aud': aud, 'iat':time.time()}, b'Bad'+goodseed, algorithm='HS256')
        resp = requests.get(f'http://localhost:{PCONF().server_config.port}/sensitive',timeout=.5,allow_redirects=False,
                           headers={'Authorization':f'Bearer {jwt_token}'})
        print(resp.text)
        assert "Failed" in resp.text
        assert resp.status_code == 401
        assert sensitive_yes_message not in resp.text

        # Try with wrong role
        jwt_token = jwt.encode({'unique_name': 'testuser', 'given_name': 'Test User', 'roles': ['not-admin'], 'aud': aud, 'iat':time.time()}, goodseed, algorithm='HS256')
        resp = requests.get(f'http://localhost:{PCONF().server_config.port}/sensitive',timeout=.5,allow_redirects=False,
                           headers={'Authorization':f'Bearer {jwt_token}'})
        print(resp.text)
        assert resp.status_code == 403
        assert resp.text == sensitive_no_message
    finally:
        server.stop()

def test_ak_api():
    
    setup_auth_test()
    from datavac.config.project_config import PCONF
    from datavac.appserve.panel_serve import launch
    from datavac.appserve.dvsecrets.ak_client_side import store_demo_access_key
    store_demo_access_key('testuser', ['admin'], other_info={'given_name': 'Test User'})
    server=launch(non_blocking=True)
    time.sleep(1)
    setup_auth_test()
    from datavac.config.project_config import PCONF
    PCONF().is_server=False; PCONF().direct_db_access=False
    try:
        from datavac.appserve.dvsecrets.ak_client_side import cli_print_user
        cli_print_user()
    finally:
        server.stop()
    
    setup_auth_test()
    from datavac.config.project_config import PCONF
    from datavac.appserve.panel_serve import launch
    from datavac.appserve.dvsecrets.ak_client_side import store_demo_access_key
    store_demo_access_key('testuser', ['admin'], other_info={'given_name': 'Test User', 'iat':0})
    MockOauthAuthorizeHandler.user_logs_in_successfully = True
    server=launch(non_blocking=True)
    time.sleep(1)
    setup_auth_test()
    from datavac.config.project_config import PCONF
    PCONF().is_server=False; PCONF().direct_db_access=False
    try:
        from datavac.appserve.dvsecrets.ak_client_side import cli_print_user, get_user
        from datavac.appserve.dvsecrets.ak_client_side import AccessKeyExpiredError
        with pytest.raises(AccessKeyExpiredError):
            get_user(client_retries=0)
        get_user(client_retries=1)
        cli_print_user()
    finally:
        server.stop()
     

if __name__ == '__main__':
    os.environ["DATAVACUUM_CONTEXT"]="builtin:demo2"
    #test_something()
    setup_auth_test()
    test_ak_download()
    setup_auth_test()
    test_ak_access()
    setup_auth_test()
    test_ak_api()