import os

import panel as pn
from pathlib import Path
from importlib import import_module

from datavac.io.database import get_database
from yaml import safe_load

import panel.theme
from panel.theme.material import MaterialDefaultTheme

from datavac.appserve.ad_auth import monkeypatch_oauthprovider
from datavac.util.logging import logger
from datavac.appserve.index import Indexer


def serve(index_yaml_file: Path = None, theme: pn.theme.Theme = MaterialDefaultTheme):

    index_yaml_file=index_yaml_file or\
                    Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"server_index.yaml"
    with open(index_yaml_file, 'r') as f:
        theyaml=safe_load(f)
        categorized_applications=theyaml['index']
        theme_module,theme_class=theyaml['theme'].split(":")
        theme=import_module(theme_module,theme_class)
        port=theyaml['port']

    kwargs={}
    possible_oauth_providers=['none','azure']
    try:
        oauth_provider=os.environ['DATAVACUUM_OAUTH_PROVIDER']
        assert oauth_provider in possible_oauth_providers
    except (KeyError,AssertionError):
        raise Exception(f"Must provide environment variable DATAVACUUM_OAUTH_PROVIDER," \
                f" options are {possible_oauth_providers}")
    if oauth_provider=='azure':
        monkeypatch_oauthprovider()
        kwargs['oauth_provider']='azure'
        kwargs['oauth_key']=os.environ['DATAVACUUM_AZURE_APP_ID']
        kwargs['oauth_secret']=os.environ['DATAVACUUM_AZURE_SSO_SECRET']
        kwargs['oauth_extra_params']={'tenant_id':os.environ['DATAVACUUM_AZURE_TENANT_ID']}
        kwargs['cookie_secret']=os.environ['DATAVACUUM_BOKEH_COOKIE_SECRET']
        kwargs['oauth_redirect_uri']=os.environ['DATAVACUUM_OAUTH_REDIRECT']
    elif oauth_provider=='none':
        logger.warning("Launching with no user authentication protocol")

    pn.state.cache['index']=index=Indexer(categorized_applications=categorized_applications)
    pn.state.cache['theme']=theme
    def authorize(user_info):

        if oauth_provider=='none': return True
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
        port=port,websocket_origin='*',show=False,
        #static_dirs={'server_static':str(ROOT_DIR/'src/datavac/server/static')},
        **kwargs)

def launch():
    db=get_database()
    db.establish_database(on_mismatch='raise')
    serve()

if __name__=='__main__':
    launch()