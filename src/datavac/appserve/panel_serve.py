import datetime
import os
from pathlib import Path
from typing import Optional

from datavac.appserve.auth import monkeypatch_authstaticroutes
from datavac.util.dvlogging import logger
from datavac.appserve.index import Indexer
from datavac.util.util import import_modfunc



def launch(index_yaml_file: Optional[Path] = None, non_blocking: bool = False):
    import panel as pn
    from datavac.config.project_config import PCONF; PCONF() # Ensure the project configuration is loaded
    sc=PCONF().server_config
    theyaml: dict = sc.get_yaml()
    additional_static_dirs={k:(v['path'],v['role']) for k,v in theyaml.get('additional_static_dirs',{}).items()}
    categorized_applications=theyaml.get('index',{})

    sc.pre_serve_setup()
    for sfunc in sc.get_scheduled_functions():
        delay = datetime.datetime.strptime(sfunc.delay,"%H:%M:%S")
        delay = datetime.timedelta(hours=delay.hour,minutes=delay.minute,seconds=delay.second)
        pn.state.schedule_task(sfunc.name, sfunc.func,
                               period=sfunc.period,
                               at=datetime.datetime.now()+delay)

    kwargs={}
    possible_oauth_providers=['none','azure','mock_azure']
    auth_info=PCONF().vault.get_auth_info()
    try:
        oauth_provider=auth_info['oauth_provider']
    except (KeyError,AssertionError):
        raise Exception(f"Must provide environment variable DATAVACUUM_OAUTH_PROVIDER," \
                f" options are {possible_oauth_providers}")
    if 'azure' in oauth_provider:
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
    else:
        raise Exception(f"Unknown oauth provider '{oauth_provider}', options are {possible_oauth_providers}")

    pn.state.cache['index']=index=Indexer(categorized_applications=categorized_applications) # type: ignore
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

    if False: pass
    #if 'shareable_secrets' in theyaml:
    #   from datavac.appserve.auth import SimpleSecretShare
    #   extra_patterns_kwargs={'extra_patterns':[
    #       ('/secretshare',SimpleSecretShare,
    #        {'callers':{k:import_modfunc(v)
    #                        for k,v in theyaml['shareable_secrets']['callers'].items()}}),
    #       ('/context',ContextDownload),
    #   ]}
    else: extra_patterns_kwargs={'extra_patterns':[]}
    from datavac.appserve.dvsecrets.ak_server_side import APIKeyGenerator
    extra_patterns_kwargs['extra_patterns'].append(('/accesskey',APIKeyGenerator))
    #index.slug_to_app['accesskey']=lambda: AccessKeyDownload().get_page()
    #index.slug_to_role['accesskey']=None#theyaml['shareable_secrets']['role']

    #extra_patterns_kwargs['extra_patterns']+=\
    #    [('/'+k,import_modfunc(v)) for k,v in theyaml.get('additional_handlers',{}).items()]
    extra_patterns_kwargs['extra_patterns']+=\
        [(k,v) for k,v in sc.get_all_handlers().items()]
    if not len(extra_patterns_kwargs['extra_patterns']): extra_patterns_kwargs={}
    #print("\n\n\n\n")
    print("Extra Patterns kwargs",extra_patterns_kwargs)
    #print("\n\n\n\n")


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
    return pn.serve(
        index.slug_to_app, # type: ignore
        #**(dict(websocket_origin='*') if PCONF().is_server else {}),
        websocket_origin=PCONF().deployment_uri.replace("https://","").replace("http://",""), 
        port=sc.port, show=False, threaded=non_blocking,
        static_dirs=additional_static_dirs, **extra_patterns_kwargs, use_xheaders=True,
        **kwargs)


if __name__=='__main__':
    print("Launching...")
    launch()