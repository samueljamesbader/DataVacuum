import argparse
import importlib.resources as irsc
import logging
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as gfg
import dotenv
import requests
import warnings
import sys

import yaml
from urllib3.exceptions import InsecureRequestWarning

from datavac import logger
from datavac.appserve.secrets import get_ssl_rootcert_for_db, get_db_connection_info
from datavac.util.conf import CONFIG
from datavac.util.util import get_resource_path, only, import_modfunc

jmp_folder=Path(os.environ['DATAVACUUM_CACHE_DIR'])/"JMP"
jmp_folder.mkdir(exist_ok=True)

def copy_in_file(filename,addin_folder,addin_id):
    with open(filename,'r') as f1:
        with open((addin_folder/filename.name),'w') as f2:
            f2.write(f1.read()\
                     .replace("%ADDINID%",addin_id) \
                     .replace("datavacuum_helper.local",addin_id)\
                     .replace("%LOCALADDINFOLDER%",str(addin_folder)))

def make_db_connect(addin_folder,addin_id, env_values):
    if (rootcertfile:=import_modfunc(CONFIG['database']['credentials']['get_ssl_rootcert_for_db'])()):
        shutil.copy(rootcertfile,addin_folder/"rootcertfile.crt")

def cli_compile_jmp_addin(*args):
    parser=argparse.ArgumentParser(description='Makes a .jmpaddin')
    #parser.add_argument('envname',nargs="*",help='Environments (ie *.env files) to use',default=[''])
    namespace=parser.parse_args(args)
    namespace.envname=[os.environ['DATAVACUUM_DEPLOYMENT_NAME']]

    for env in namespace.envname:
        if True:
        #if env != '':
        #    dotenv_path=dotenv.find_dotenv(f".{env}.env")
        #    assert dotenv_path, f"Didn't find .{env}.env"
        #    env_values=dotenv.dotenv_values(dotenv_path)
        #else:
            env_values = os.environ
        envname=(env if len(env) else "LOCAL")
        addin_id=f'datavacuum_helper.{envname.lower()}'
        print(f"Using {envname}")

        if (jmp_conf:=Path(env_values["DATAVACUUM_CONFIG_DIR"])/"jmp.yaml").exists():
            with open(jmp_conf,'r') as f:
                jmp_conf=yaml.safe_load(f)
        else: jmp_conf={}

        addin_folder=jmp_folder/envname
        if addin_folder.exists():
            shutil.rmtree(addin_folder)
        addin_folder.mkdir(exist_ok=False)

        with open(addin_folder/"Addin.def",'w') as f:
            f.writelines([
                f"id={addin_id}\n",
                f"name=DataVac_{envname.capitalize()}\n",
                "supportJmpSE=0\n",
                "addinVersion=1\n",
                "minJmpVersion=16",])

        with open(addin_folder/"env_vars.jsl",'w') as f:
            print(sys.path)
            potential_dlls=sum((list(Path(p).glob("Python31*.dll")) for p in sys.path),[])
            # TODO: Remove SSLROOTCERT info here, rely on rootcertfile existing or not
            built_in_capture_vars={'DATAVACUUM_SSLROOTCERT':(str(get_ssl_rootcert_for_db()) or ''),
                                   'PYTHON_SYS_PATHS': ";".join(sys.path),
                                   'PYTHON_DLL':str(potential_dlls[0])}
            # TODO: Remove DB connection info from JMP add-in, rely on shareable secrets
            connection_info=get_db_connection_info()

            f.write("Names Default To Here(1);\n")
            f.write(f"dv = Namespace(\"{addin_id}\");\n")
            for varname, varvalue in dict(**jmp_conf.get("capture_variables",{}),**built_in_capture_vars).items():
                if len(varvalue) and varvalue[0]=="%" and varvalue[-1]=="%":
                    varvalue=env_values[varvalue[1:-1]]
                f.write(f"dv:{varname}=\"{varvalue}\";\n")

            for varname, connstrname in [('DATABASE','Database'),('SERVER','Server'),('PORT','Port'),
                                         ('UID','Uid'),('PWD','Password')]:
                f.write(f"dv:DATAVACUUM_DB_{varname}=\"{connection_info[connstrname]}\";\n")

        with open(addin_folder/"jmp16_pyinit.py",'w') as f:

            # This generates an add-on that will only work for the user who compiles it
            # but I'm not going to invest in doing this portably now because it will all change
            # when JMP 18 comes out with a complete re-write of Python integration...
            #f.write("import sys\n")
            f.write("import os\n")
            #f.write("print('yo');")
            #f.write("paths=[r'"+"',r'".join(sys.path)+"']\n")
            #f.write("print(paths);")
            #f.write("[sys.path.append(path) for path in paths if path not in sys.path]\n")
            for x in ['DATAVACUUM_CONFIG_DIR','DATAVACUUM_DB_DRIVERNAME',
                      'DATAVACUUM_DBSTRING','DATAVACUUM_CACHE_DIR','DATAVACUUM_LAYOUT_PARAMS_DIR']:
                f.write(f"os.environ['{x}']=r'{env_values.get(x,None)}'\n" if env_values.get(x,None) else "")
            f.write("import numpy as np\n")
            f.write("import pandas as pd\n")
            f.write("from datavac.io.database import get_database; db = get_database()")
            #f.write("print(np.r_[1,2])\n")
            #f.write("import datavac\n")

        with open(addin_folder/"addinLoad.jsl",'w') as f:
            generated_jsl=[addin_folder/'env_vars.jsl']
            dv_base_jsl=[get_resource_path(x) for x in ['datavac.jmp:ConnectToWaferMap.jsl',
                                                        'datavac.jmp:DBConnect.jsl',
                                                        'datavac.jmp:JMP16Python.jsl',
                                                        'datavac.jmp:Util.jsl',
                                                        *(['datavac.jmp:ReloadAddin.jsl'] if envname=='LOCAL' else [])]]
            request_jsl=[get_resource_path(x) for x in jmp_conf.get('additional_jsl',[])]
            inc_files=[*generated_jsl,*dv_base_jsl,*request_jsl]
            inc_filenames=[Path(inc_file).name for inc_file in inc_files]
            assert len(set(inc_filenames))==len(inc_filenames), \
                f"Repeated filename in {inc_files}"
            f.writelines([
                f'Names Default To Here(1);\n',
                f'dv=New Namespace("{addin_id}");\n',
                *[f'Include( "$ADDIN_HOME({addin_id})/{Path(filename).name}" );\n'
                    for filename in inc_files],
                f'dv:addin_home=Get Path Variable("ADDIN_HOME({addin_id})")'
            ])
            for add_file in [*dv_base_jsl,*request_jsl]:
                copy_in_file(add_file,addin_folder=addin_folder,addin_id=addin_id)

        commands=[
            {
                'name':'Connect to Wafermap',
                'tip':'Assign map role for current table',
                'text': f'dv=Namespace("{addin_id}");dv:ConnectToWafermap();',
                'icon':None
            },
            *([{
                'name':'Reload Addin',
                'tip':'Reload this add-in',
                'text': f'dv=Namespace("{addin_id}");dv:ReloadAddin();',
                'icon':None
            }] if envname=="LOCAL" else []),
            {
                'name':'Pull Sweeps',
                'tip':'Pull raw curves corresponding to open table',
                'text': f'dv=Namespace("{addin_id}");dv:PullSweeps();',
                'icon':None
            },
            {
                'name':'Abs Currents',
                'tip':'For headers that look like currents, take absolute value',
                'text': f'dv=Namespace("{addin_id}");dv:AbsCurrents();',
                'icon':None
            },
        ]
        with (open(addin_folder/"addin.jmpcust",'wb') as f):
            root=gfg.Element("jm:menu_and_toolbar_customizations")
            root.set("xmlns:jm","http://www.jmp.com/ns/menu")
            root.set("version","3")
            root.append((iimm:=gfg.Element("jm:insert_in_main_menu")))
            iimm.append((iim:=gfg.Element("jm:insert_in_menu")))
            iim.append((main_menu:=gfg.Element("jm:name")))
            main_menu.text='ADD-INS'
            iim.append((ia:=gfg.Element("jm:insert_after")))
            ia.append(gfg.Element("jm:name"))
            ia.append((dvmenu:=gfg.Element("jm:menu")))
            dvmenu.append((dvmenu_name:=gfg.Element("jm:name")))
            dvmenu_name.text=f'DataVac_{envname.capitalize()}'
            dvmenu.append((dvmenu_caption:=gfg.Element("jm:caption")))
            dvmenu_caption.text=f'DataVac_{envname.capitalize()}'
            for command in commands:
                dvmenu.append((comm:=gfg.Element("jm:command")))
                comm.append((comm_name:=gfg.Element("jm:name")))
                comm_name.text=command['name']
                comm.append((comm_cap:=gfg.Element("jm:caption")))
                comm_cap.text=command['name']
                comm.append((comm_act:=gfg.Element("jm:action")))
                #comm_act.set("type","path")
                #comm_act.text=fr"$ADDIN_HOME({addin_id})\{command['filename']}"
                comm_act.text=command['text']
                comm_act.set("type","text")
                comm.append((comm_tip:=gfg.Element("jm:tip")))
                comm_tip.text=command['tip']
                assert command['icon'] is None
                comm.append((comm_icon:=gfg.Element("jm:icon")))
                comm_icon.set('type','None')

                #copy_in_file(command['filename'],addin_folder,addin_id=addin_id)

            tree=gfg.ElementTree(root)
            gfg.indent(tree,'  ')
            tree.write(f)

            make_db_connect(addin_folder=addin_folder,addin_id=addin_id,env_values=env_values)

        shutil.make_archive(addin_folder/f"DataVac_{envname.capitalize()}",'zip', addin_folder)
        shutil.move(addin_folder/f"DataVac_{envname.capitalize()}.zip",addin_folder/f"DataVac_{envname.capitalize()}.jmpaddin")
