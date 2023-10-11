import os

import panel as pn
from pathlib import Path

import panel.theme
from panel.theme.material import MaterialDefaultTheme

from .ad_auth import monkeypatch_oauthprovider
from datavac.logging import logger
from .index import Indexer


def serve(index_yaml_file: Path, oauth='azure', theme: pn.theme.Theme = MaterialDefaultTheme):

    kwargs={}
    if oauth:
        monkeypatch_oauthprovider()
        #kwargs['auth_provider']=AuthModule(ROOT_DIR/"src/datavac/server/password_auth.py")
        kwargs['oauth_provider']='azure'
        kwargs['oauth_key']=os.environ['DATAVACUUM_AZURE_APP_ID']
        kwargs['oauth_secret']=os.environ['DATAVACUUM_AZURE_SSO_SECRET']
        kwargs['oauth_extra_params']={'tenant_id':os.environ['DATAVACUUM_AZURE_TENANT_ID']}
        kwargs['cookie_secret']=os.environ['DATAVACUUM_BOKEH_COOKIE_SECRET']
        kwargs['oauth_redirect_uri']=os.environ['DATAVACUUM_OAUTH_REDIRECT']

    pn.state.cache['index']=index=Indexer(index_yaml_file=index_yaml_file)
    pn.state.cache['theme']=theme
    def authorize(user_info):

        if not oauth: return True
        try:
            uri=pn.state.curdoc.session_context._request.uri
            assert uri[0]=="/", f"URI '{uri}' is not as expected"
            slug=uri[1:].split("?")[0]

            if user_info:
                logger.info(f"Login attempt to '{slug}'")
                if (required_role:=index.slug_to_role[slug]) in user_info.get('roles',[]):
                    logger.info(f"User has role '{required_role}', accepting login.")
                    return True
                else:
                    logger.info(f"User does not have role '{required_role}', rejecting login.")
                    return False
            else:
                logger.info(f"Non-login")
                return False

        except Exception as e:
            logger.warning("Failure in authorization code")
            logger.warning(str(e))
            return False

    pn.config.authorize_callback = authorize
    pn.serve(
        index.slug_to_app,
        port=8080,websocket_origin='*',show=False,
        #static_dirs={'server_static':str(ROOT_DIR/'src/datavac/server/static')},
        **kwargs)
