import argparse
import os
from pathlib import Path
import shutil
from textwrap import dedent
import sys
import yaml

from datavac.util.dvlogging import logger
from datavac.util.util import get_resource_path


def copy_in_file(filename,addin_folder,addin_id):
    with open(filename,'r') as f1:
        f1lines=f1.readlines()
        assert f1lines[0].replace(" ","").strip()=='NamesDefaultToHere(1);dv=:::dv;',\
            f"All JSL files to copy into add-in must start with 'Names Default To Here(1); dv=:::dv;'.  See {filename}."
        f1content="".join(['Names Default To Here(1);\ndv=Namespace("%ADDINID%");\n',*f1lines[1:]])
        with open((addin_folder/filename.name),'w') as f2:
            f2.write(f1content\
                     .replace("%ADDINID%",addin_id) \
                     #.replace("datavacuum_helper.local",addin_id)\
                     .replace("%LOCALADDINFOLDER%",str(addin_folder)))

def make_db_connect(addin_folder,addin_id, env_values):
    from datavac.config.project_config import PCONF
    if (rootcertfile:=PCONF().cert_depo.get_ssl_rootcert_path_for_db()):
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
        #print(f"Using {envname}")
        from datavac.config.project_config import PCONF
        CONFIG_DIR=PCONF().CONFIG_DIR
        assert CONFIG_DIR is not None, "No project configuration directory, are you in the right context?"
        if (jmp_conf:=Path(CONFIG_DIR/"jmp.yaml")).exists():
            with open(jmp_conf,'r') as f:
                jmp_conf=yaml.safe_load(f)
        else: jmp_conf={}

        jmp_folder=PCONF().USER_CACHE/"JMP"
        jmp_folder.mkdir(exist_ok=True)
        addin_folder=jmp_folder
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
            #print(sys.path)
            potential_dlls=sum((list(Path(p).glob("Python31*.dll")) for p in sys.path),[])
            built_in_capture_vars={'DATAVACUUM_DEPLOYMENT_URI':env_values.get("DATAVACUUM_DEPLOYMENT_URI",""),
                                   'PYTHON_SYS_PATHS': ";".join(sys.path),
                                   'PYTHON_DLL':str(potential_dlls[0]),
                                   'DATAVACUUM_JMP_DEFER_INIT':env_values.get("DATAVACUUM_JMP_DEFER_INIT","NO"),
                                  }

            f.write("Names Default To Here(1);\n")
            f.write(f"dv = Namespace(\"{addin_id}\");\n")
            for varname, varvalue in dict(**jmp_conf.get("capture_variables",{}),**built_in_capture_vars).items():
                if len(varvalue) and varvalue[0]=="%" and varvalue[-1]=="%":
                    varvalue=env_values[varvalue[1:-1]]
                f.write(f"dv:{varname}=\"{varvalue}\";\n")


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
            #for x in ['DATAVACUUM_CONFIG_DIR','DATAVACUUM_DB_DRIVERNAME',
            #          'DATAVACUUM_DBSTRING','DATAVACUUM_CACHE_DIR','DATAVACUUM_LAYOUT_PARAMS_DIR']:
            for x in ['DATAVACUUM_CONTEXT','DATAVACUUM_CONTEXT_DIR','DATAVACUUM_DB_DRIVERNAME',
                      'DATAVACUUM_JMP_DEFER_INIT',*jmp_conf.get("capture_variables",{})]:
                f.write(f"os.environ['{x}']=r\"{env_values.get(x,None)}\"\n" if env_values.get(x,None) else "")
            f.write(dedent("""
                os.environ['DATAVACUUM_FROM_JMP']='YES'
                for k,v in os.environ.items():
                    if 'DATAVAC' in k:
                        print(f"{k.ljust(35)}={v}")
                import numpy as np
                import pandas as pd
                from datavac.database.db_get import get_sweeps_for_jmp
            """))
            #f.write("print(np.r_[1,2])\n")
            #f.write("import datavac\n")

        with open(addin_folder/"addinLoad.jsl",'w') as f:
            generated_jsl=[]#addin_folder/'env_vars.jsl']
            dv_base_jsl=[get_resource_path(x) for x in [
                                                        'datavac.jmp::JMP16Python.jsl',
                                                        'datavac.jmp::Secrets.jsl',
                                                        'datavac.jmp::DBConnect.jsl',
                                                        'datavac.jmp::Util.jsl',
                                                        'datavac.jmp::ConnectToWaferMap.jsl',
                                                        'datavac.jmp::ReloadAddin.jsl',
                                                        'datavac.jmp::SplitTables.jsl',
                                                        #*(['datavac.jmp::ReloadAddin.jsl'] if envname=='LOCAL' else [])
                                                       ]]
            request_jsl=[get_resource_path(x) for x in jmp_conf.get('additional_jsl',[])]
            inc_files=[*generated_jsl,*dv_base_jsl,*request_jsl]
            inc_filenames=[Path(inc_file).name for inc_file in inc_files]
            assert len(set(inc_filenames))==len(inc_filenames), \
                f"Repeated filename in {inc_files}"
            f.writelines([
                f'Names Default To Here(1);\n',
                f'dv=Namespace("{addin_id}");\n',
                f'If(Namespace Exists(dv)&(!IsEmpty(dv:force_init)),force_init=dv:force_init,force_init=0);\n',
                f'dv=New Namespace("{addin_id}");\n',
                f'dv:name="{envname}";\n',
                f'dv:addin_home=Get Path Variable("ADDIN_HOME({addin_id})");\n',
                f'Include( "$ADDIN_HOME({addin_id})/env_vars.jsl" );\n',
                f'If((dv:DATAVACUUM_JMP_DEFER_INIT!="YES")|force_init,\n',
                # Using :::dv so that add-ins work inside of JMP projects
                # See https://community.jmp.com/t5/Discussions/Unable-to-run-addins-within-Project/td-p/383815
                f':::dv=dv;\n',
                *[f'  Include( "$ADDIN_HOME({addin_id})/{Path(filename).name}" );\n'
                    for filename in inc_files],
                f'  dv:force_init=0;\n',
                f',//Else\n  Write("Deferred initialization of {addin_id} because of DATAVACUUM_JMP_DEFER_INIT\\!N"));'
            ])
            for add_file in [*dv_base_jsl,*request_jsl]:
                copy_in_file(add_file,addin_folder=addin_folder,addin_id=addin_id)

        general_commands=[
            {
                'name':'Connect to Wafermap',
                'tip':'Assign map role for current table',
                'text': f'dv:ConnectToWafermap();',
                'icon':None
            },
            *([{
                'name':'Reload Addin',
                'tip':'Reload this add-in',
                'text': f'dv:ReloadAddin();',
                'icon':None
            }] ),#if envname=="LOCAL" else []),
            {
                'name':'Pull Sweeps',
                'tip':'Pull raw curves corresponding to open table',
                'text': f'dv:PullSweeps();',
                'icon':None
            },
            {
                'name':'Abs Currents',
                'tip':'For headers that look like currents, take absolute value',
                'text': f'dv:AbsCurrents();',
                'icon':None
            },
            {
                'name':'Attach Splits',
                'tip':'Attach to a split table',
                'text': f'dv:AttachSplitTable();',
                'icon':None
            },
        ]
        menus=[
            *([{
                'name':'Init',
                'tip':'Initialize the add-in',
                'text': f'dv:force_init=1;Include( dv:addin_home||"/addinLoad.jsl");',
                'icon':None
            }] if env_values.get("DATAVACUUM_JMP_DEFER_INIT","NO")=='YES' else []),
            {'General':general_commands},*jmp_conf.get('menus',[])]
        with (open(addin_folder/"addin.jmpcust",'wb') as f):
            import xml.etree.ElementTree as gfg
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
            def populate_menu(menu,items):
                for item in items:
                    assert type(item) is dict
                    if 'name' in item:
                        command=item
                        menu.append((comm:=gfg.Element("jm:command")))
                        comm.append((comm_name:=gfg.Element("jm:name")))
                        comm_name.text=command['name']
                        comm.append((comm_cap:=gfg.Element("jm:caption")))
                        comm_cap.text=command['name']
                        comm.append((comm_act:=gfg.Element("jm:action")))
                        text=f'dv=Namespace("{addin_id}");'
                        for inc in command.get('includes',[]):
                            text+=f'Include("{get_resource_path(inc)}");'
                        text+=command['text']
                        comm_act.text=text
                        comm_act.set("type","text")
                        comm.append((comm_tip:=gfg.Element("jm:tip")))
                        comm_tip.text=command.get('tip',command['name'])
                        assert command.get('icon',None) is None
                        comm.append((comm_icon:=gfg.Element("jm:icon")))
                        comm_icon.set('type','None')
                    else:
                        assert len(item)==1
                        itemname,itemcontent=list(item.items())[0]
                        submenu=itemcontent
                        menu.append((sub:=gfg.Element("jm:menu")))
                        sub.append((sub_name:=gfg.Element("jm:name")))
                        sub.append((sub_cap:=gfg.Element("jm:caption")))
                        sub_cap.text=itemname
                        sub_name.text=itemname
                        populate_menu(sub,submenu)
            populate_menu(dvmenu,menus)

            tree=gfg.ElementTree(root)
            gfg.indent(tree,'  ')
            tree.write(f)

            make_db_connect(addin_folder=addin_folder,addin_id=addin_id,env_values=env_values)

        shutil.make_archive(str(addin_folder/f"DataVac_{envname.capitalize()}"),'zip', addin_folder)
        shutil.move(addin_folder/f"DataVac_{envname.capitalize()}.zip",addin_folder/f"DataVac_{envname.capitalize()}.jmpaddin")

        logger.debug(f"Add-in for {envname} compiled")
        print("\n\nNow install by opening the following file in JMP:")
        print(fr"{addin_folder/f'DataVac_{envname.capitalize()}.jmpaddin'}")
        print("")