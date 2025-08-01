import datetime
import logging
import os
from pathlib import Path
from typing import Optional
from yaml import safe_load

import panel as pn
#from panel.theme import Theme
from panel.theme.material import MaterialDefaultTheme

from datavac.appserve.ad_auth import monkeypatch_oauthprovider, monkeypatch_authstaticroutes, AccessKeyDownload
from datavac.util.dvlogging import logger
from datavac.appserve.index import Indexer
from datavac.util.util import import_modfunc



def launch(index_yaml_file: Optional[Path] = None):

    from datavac.config.project_config import PCONF; PCONF() # Ensure the project configuration is loaded
    index_yaml_file=index_yaml_file or\
                    Path(os.environ['DATAVACUUM_CONFIG_PATH']).parent/"server_index.yaml"
    with open(index_yaml_file, 'r') as f:
        f=f.read()
        for k,v in os.environ.items():
            if 'DATAVAC' in k: f=f.replace(f"%{k}%",v)
        theyaml=safe_load(f)
        additional_static_dirs={k:(v['path'],v['role']) for k,v in theyaml.get('additional_static_dirs',{}).items()}
        categorized_applications=theyaml['index']
        theme=import_modfunc(theyaml['theme']) if 'theme' in theyaml else MaterialDefaultTheme
        port=theyaml.get('port',3000)


    for sf in theyaml.get('setup_functions',[]):
        import_modfunc(sf)()
    for sf,sfi in theyaml.get('scheduled_functions',{}).items():
        delay = datetime.datetime.strptime(sfi['delay'],"%H:%M:%S")
        delay = datetime.timedelta(hours=delay.hour,minutes=delay.minute,seconds=delay.second)
        pn.state.schedule_task(sf, import_modfunc(sfi['func']),
                               period=sfi['period'],
                               at=datetime.datetime.now()+delay)
        #logger.debug("Thing that needs KRB5")


    kwargs={}
    possible_oauth_providers=['none','azure']
    # TODO: make this use dvsecrets get_auth_info
    auth_info=PCONF().vault.get_auth_info()
    try:
        oauth_provider=auth_info['oauth_provider']
        assert oauth_provider in possible_oauth_providers
    except (KeyError,AssertionError):
        raise Exception(f"Must provide environment variable DATAVACUUM_OAUTH_PROVIDER," \
                f" options are {possible_oauth_providers}")
    if oauth_provider=='azure':
        monkeypatch_oauthprovider()
        for k in ['oauth_provider','oauth_key','oauth_secret',
                  'oauth_extra_params','oauth_redirect_uri','cookie_secret']:
            kwargs[k]=auth_info[k]
    elif oauth_provider=='none':
        if os.environ.get('DATAVACUUM_PASSWORD'):
            logger.warning("Launching with password authentication, NOT MEANT FOR PRODUCTION!")
            kwargs['cookie_secret']=os.environ['DATAVACUUM_BOKEH_COOKIE_SECRET']
            kwargs['basic_auth']=os.environ['DATAVACUUM_PASSWORD']
        else:
            logger.warning("Launching with NO user authentication protocol!")

    pn.state.cache['index']=index=Indexer(categorized_applications=categorized_applications) # type: ignore
    pn.state.cache['theme']=theme # type: ignore
    def authorize(user_info,request_path):
        if oauth_provider=='none': return True
        try:
            #uri=pn.state.curdoc.session_context._request.uri
            #print(uri,request_path)
            uri=request_path
            assert uri[0]=="/", f"URI '{uri}' is not as expected"
            slug=uri[1:].split("?")[0]

            if user_info:
                logger.info(f"Login attempt to '{slug}'")
                if (required_role:=index.slug_to_role[slug]) in [None,'None']:
                    logger.info(f"Role not required for '{slug}', accepting login.")
                    return True
                elif required_role in user_info.get('roles',[]):
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

    monkeypatch_authstaticroutes()
    pn.config.authorize_callback = authorize

    if 'shareable_secrets' in theyaml:
        from datavac.appserve.ad_auth import SimpleSecretShare
        extra_patterns_kwargs={'extra_patterns':[
            ('/secretshare',SimpleSecretShare,
             {'callers':{k:import_modfunc(v)
                             for k,v in theyaml['shareable_secrets']['callers'].items()}}),
            ('/context',ContextDownload),
        ]}
        index.slug_to_app['accesskey']=lambda: AccessKeyDownload().get_page()
        index.slug_to_role['accesskey']=theyaml['shareable_secrets']['role']
    else: extra_patterns_kwargs={'extra_patterns':[]}

    extra_patterns_kwargs['extra_patterns']+=\
        [('/'+k,import_modfunc(v)) for k,v in theyaml.get('additional_handlers',{}).items()]
    if not len(extra_patterns_kwargs['extra_patterns']): extra_patterns_kwargs={}
    print("\n\n\n\n")
    print("Extra Patterns kwargs",extra_patterns_kwargs)
    print("\n\n\n\n")


    #def alter_logs():
    #    print("altering logs")
    #    #blog=logging.getLogger(name="bokeh")
    #    #blog.setLevel("INFO")
    #    #blog=logging.getLogger(name="panel")
    #    #blog.setLevel("DEBUG")
    #    for bname in ['tornado.access','auth']:
    #        blog=logging.getLogger(name=bname)
    #        blog.setLevel(logging.DEBUG)
    #        import sys
    #        ch=logging.StreamHandler(sys.stdout)
    #        ch.setLevel(logging.DEBUG)
    #        blog.addHandler(ch)

    #alter_logs()
    #pn.state.schedule_task('altlogs',alter_logs,
    #                       period='10m')
    # use_xheaders=True: https://docs.bokeh.org/en/2.4.2/docs/user_guide/server.html#reverse-proxying-with-nginx-and-ssl
    pn.serve(
        index.slug_to_app, # type: ignore
        port=port,websocket_origin='*',show=False,
        static_dirs=additional_static_dirs, **extra_patterns_kwargs, use_xheaders=True,
        **kwargs)

from tornado.web import RequestHandler
class ContextDownload(RequestHandler):
    def get(self):
        depname=os.environ['DATAVACUUM_DEPLOYMENT_NAME']
        self.set_header('Content-Disposition', f'attachment; filename={depname}.dvcontext.env')
        self.write(f"# Context file for '{depname}'\n")
        self.write(f"# Downloaded {datetime.datetime.now()}\n")
        for name in ['DATAVACUUM_DEPLOYMENT_NAME','DATAVACUUM_DEPLOYMENT_URI','DATAVACUUM_CONFIG_MODULE']:
            self.write(f"{name}={os.environ[name]}\n")

if __name__=='__main__':
    launch()