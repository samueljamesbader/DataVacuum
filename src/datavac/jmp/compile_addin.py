import argparse
import os
from pathlib import Path
import shutil
from textwrap import dedent
import sys
import yaml

from datavac.util.dvlogging import logger
from datavac.util.util import get_resource_path

default_inside=[
    'datavac.jmp::JMP16Python.jsl',
    'datavac.jmp::DBConnect.jsl',
    'datavac.jmp::Util.jsl',
    'datavac.jmp::ConnectToWafermap.jsl',
    'datavac.jmp::ReloadAddin.jsl',
    'datavac.jmp::SplitTables.jsl',

    'ScriptViewer.jsl',
]

def copy_in_file(filename,addin_folder,addin_id):
    with open(filename,'r') as f1:
        f1lines=f1.readlines()
        assert next(l for l in f1lines if len(l.strip()) and not l.strip().startswith("//")).replace(" ","").strip()=='NamesDefaultToHere(1);dv=:::dv;',\
            f"All JSL files to copy into add-in must start with 'Names Default To Here(1); dv=:::dv;'.  See {filename}."
        f1content="".join(['Names Default To Here(1);\ndv=Namespace("%ADDINID%");\n',*f1lines[1:]])
        with open((addin_folder/filename.name),'w') as f2:
            f2.write(f1content\
                     .replace("%ADDINID%",addin_id) \
                     #.replace("datavacuum_helper.local",addin_id)\
                     .replace("%LOCALADDINFOLDER%",str(addin_folder)))

def get_jsl_path(inc, addin_id, jmp_conf):
    if inc in jmp_conf.get('to_copy_in',default_inside):
        if '::' in inc: return f"$ADDIN_HOME({addin_id})/{get_resource_path(inc).name}"
        else: return f"$ADDIN_HOME({addin_id})/{inc}"
    else: return get_resource_path(inc)

def make_db_connect(addin_folder,addin_id, env_values):
    from datavac.config.project_config import PCONF
    if (rootcertfile:=PCONF().cert_depo.get_ssl_rootcert_path_for_db()):
        shutil.copy(rootcertfile,addin_folder/"rootcertfile.crt")


def make_addin_def(addin_folder, addin_id, envname):
    with open(addin_folder/"Addin.def",'w') as f:
        f.writelines([
            f"id={addin_id}\n",
            f"name=DataVac_{envname.capitalize()}\n",
            "supportJmpSE=0\n",
            "addinVersion=1\n",
            "minJmpVersion=16",])

def make_addin_load(addin_folder, addin_id, envname, jmp_conf, menu_includes={}):
    with open(addin_folder/"addinLoad.jsl",'w') as f:
        inside_jsl_dotpaths=jmp_conf.get('to_copy_in',default_inside)
        all_jsl_dotpaths=list(dict.fromkeys(inside_jsl_dotpaths+jmp_conf.get('additional_jsl',[])+list(menu_includes.values())))
        assert len([x for x in all_jsl_dotpaths])==len(set([x.split("::")[-1].lower() for x in all_jsl_dotpaths])),\
            "Repeated JSL filename with different dot path among "+str(list(all_jsl_dotpaths))
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
            f'  :::dv=dv;\n',
            *[f'  Include( "{get_jsl_path(inc,addin_id,jmp_conf)}" );\n' for inc in all_jsl_dotpaths],
            #*[f'  Include( "{inc}");\n' for inc in menu_jsl],
            f'  dv:force_init=0;\n',
            f',//Else\n  Write("Deferred initialization of {addin_id} because of DATAVACUUM_JMP_DEFER_INIT\\!N"));'
        ])
        for add_file in inside_jsl_dotpaths:
            if '::' in add_file:
                copy_in_file(get_resource_path(add_file),addin_folder=addin_folder,addin_id=addin_id)

def make_env_vars(addin_folder, addin_id, jmp_conf, env_values):
    from datavac.config.project_config import PCONF
    import sys
    from pathlib import Path
    with open(addin_folder/"env_vars.jsl",'w') as f:
        potential_dlls = sum((list(Path(p).glob("Python31*.dll")) for p in sys.path), [])
        built_in_capture_vars: dict[str, str] = {
            'DATAVACUUM_DEPLOYMENT_URI': PCONF().deployment_uri,
            'PYTHON_SYS_PATHS': ";".join(sys.path),
            'PYTHON_DLL': str(potential_dlls[0]) if potential_dlls else "",
            'DATAVACUUM_JMP_DEFER_INIT': env_values.get("DATAVACUUM_JMP_DEFER_INIT", "NO"),
            'DATAVACUUM_DIRECT_DB_ACCESS': env_values.get("DATAVACUUM_DIRECT_DB_ACCESS", "NO"),
        }

        f.write("Names Default To Here(1);\n")
        f.write(f"dv = Namespace(\"{addin_id}\");\n")
        for varname, varvalue in dict(**jmp_conf.get("capture_variables", {}), **built_in_capture_vars).items():
            try:
                if len(varvalue) and varvalue[0] == "%" and varvalue[-1] == "%":
                    varvalue = env_values[varvalue[1:-1]]
                f.write(f"dv:{varname}=\"{varvalue}\";\n")
            except Exception as e:
                logger.debug(f"Skipping {varname} because {e}")

def make_pyinit(addin_folder, addin_id, jmp_conf, env_values):
    import sys
    from textwrap import dedent
    with open(addin_folder/"jmp16_pyinit.py",'w') as f:
        f.write("import os\n")
        for x in [
            'DATAVACUUM_CONTEXT','DATAVACUUM_CONTEXT_DIR','DATAVACUUM_DB_DRIVERNAME',
            'DATAVACUUM_JMP_DEFER_INIT','DATAVACUUM_DIRECT_DB_ACCESS','DATAVACUUM_READ_DIR',
            *jmp_conf.get("capture_variables",{})
        ]:
            f.write(f"os.environ['{x}']=r\"{env_values.get(x,None)}\"\n" if env_values.get(x,None) else "")
        f.write(dedent("""
            os.environ['DATAVACUUM_FROM_JMP']='YES'
            for k,v in os.environ.items():
                if 'DATAV' in k:
                    print(f"{k.ljust(35)}={v}")
            import numpy as np
            import pandas as pd
            from datavac.database.db_get import get_data, get_factors, get_sweeps_for_jmp
        """))

def make_addin_jmp_cust(addin_folder, addin_id, envname, jmp_conf, env_values):
    includes={}
    NOCODE=(os.environ.get("DATAVACUUM_NOCODE","False").capitalize()=='True')
    general_commands=[
        {
            'name':'Connect to Wafermap',
            'tip':'Assign map role for current table',
            'text': f'dv=:::dv;dv:ConnectToWafermap();',
            'includes': ['datavac.jmp::ConnectToWafermap.jsl'],
            'icon':None
        },
        ({
            'name':'Refetch Addin',
            'tip':'Refetch and reload this add-in',
            'text': f'dv=:::dv;dv:RefetchAddin();',
            'icon':None
        } if NOCODE else
        {
            'name':'Reload Addin',
            'tip':'Reload this add-in',
            'text': f'dv=:::dv;dv:ReloadAddin();',
            'icon':None
        }),
        {
            'name':'Login/Re-login',
            'tip':'Logout and login again to refresh access key',
            'text': f'dv=:::dv;dv:ReLogin();',
            'icon':None
        },
        #{
        #    'name':'Pull Sweeps',
        #    'tip':'Pull raw curves corresponding to open table',
        #    'text': f'dv:PullSweeps();',
        #    'icon':None
        #},
        {
            'name':'Abs Currents',
            'tip':'For headers that look like currents, take absolute value',
            'text': f'dv=:::dv;dv:AbsCurrents();',
            'includes': ['datavac.jmp::Util.jsl'],
            'icon':None
        },
        {
            'name':'Attach Splits',
            'tip':'Attach to a split table',
            'text': f'dv=:::dv;dv:AttachSplitTable();',
            'includes': ['datavac.jmp::SplitTables.jsl'],
            'icon':None
        },
        {
            'name':'Get Data',
            'tip':'Get data from DataVacuum',
            'text': f'dv=:::dv;dv:GetDataWithLotGui("?","?");',
            'includes': ['datavac.jmp::Util.jsl'],
            'icon':None
        },
        {
            'name':'Script Viewer',
            'tip':'View source JSL scripts',
            'text': f'dv=:::dv;dv:ScriptViewer();',
            #'includes': ['datavac.jmp::ScriptViewer.jsl'],
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
        def populate_menu(menu,items,superm:str|None):
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
                        text+=f'Include( "{get_jsl_path(inc, addin_id, jmp_conf)}");'
                        includes[(superm+"->"+command['name'] if superm else command['name'])+\
                                 " ("+Path(get_jsl_path(inc,addin_id,jmp_conf)).name+")"]=inc
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
                    populate_menu(sub,submenu,superm=superm+"->"+itemname if superm else itemname)
        populate_menu(dvmenu,menus, superm=None)

        tree=gfg.ElementTree(root)
        gfg.indent(tree,'  ')
        tree.write(f)
        return includes

def make_script_viewer(addin_folder, addin_id, menu_includes:dict[str,str], jmp_conf):
    with open(get_resource_path('datavac.jmp::ScriptViewer.jsl'),'r') as f1:
        f1content=f1.read()
    with open((addin_folder/"ScriptViewer.jsl"),'w') as f2:
        f2.write(f1content\
                 .replace("%ADDINID%",addin_id) \
                 .replace("%LOCALADDINFOLDER%",str(addin_folder))\
                 .replace("%DEFINEARRAYSHERE%", "{\n"+",\n".join(f'   {{"{k}" , "{get_jsl_path(v,addin_id,jmp_conf)}"}}' for k,v in menu_includes.items())+"\n}")
                 )

def cli_compile_jmp_addin(*args):
    parser=argparse.ArgumentParser(description='Makes a .jmpaddin')
    parser.add_argument('--attempt_install', action='store_true', help='Attempt to open the add-in with JMP after building')
    #parser.add_argument('envname',nargs="*",help='Environments (ie *.env files) to use',default=[''])
    namespace=parser.parse_args(args)
    namespace.envname=[os.environ['DATAVACUUM_DEPLOYMENT_NAME']]

    env=os.environ['DATAVACUUM_DEPLOYMENT_NAME']
    env_values = os.environ
    envname=(env if len(env) else "LOCAL")
    addin_id=f'datavacuum_helper.{envname.lower()}'
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

    menu_includes=make_addin_jmp_cust(addin_folder, addin_id, envname, jmp_conf, env_values)
    make_addin_def(addin_folder, addin_id, envname)
    make_env_vars(addin_folder, addin_id, jmp_conf, env_values)
    make_pyinit(addin_folder, addin_id, jmp_conf, env_values)
    make_addin_load(addin_folder, addin_id, envname, jmp_conf, menu_includes=menu_includes)
    make_db_connect(addin_folder=addin_folder,addin_id=addin_id,env_values=env_values)
    make_script_viewer(addin_folder, addin_id, menu_includes=menu_includes, jmp_conf=jmp_conf)

    shutil.make_archive(str(addin_folder/f"DataVac_{envname.capitalize()}"),'zip', addin_folder)
    shutil.move(addin_folder/f"DataVac_{envname.capitalize()}.zip",addin_folder/f"DataVac_{envname.capitalize()}.jmpaddin")

    jmp_paths=[
        Path(r"C:\Program Files\SAS\JMPPRO\17\Jmp.exe"),
        Path(r"C:\Program Files\SAS\JMPPRO\16\Jmp.exe")
    ]
    path_to_open=fr"{addin_folder/f'DataVac_{envname.capitalize()}.jmpaddin'}"

    logger.debug(f"Add-in for {envname} compiled")
    if getattr(namespace, "attempt_install", False):
        jmp_path=next((p for p in jmp_paths if p.exists()),None)
        if jmp_path is not None:
            import subprocess
            try:
                subprocess.Popen([str(jmp_path), str(path_to_open)])
                print(f"Attempted to open {path_to_open} with JMP at {jmp_path}")
            except Exception as e:
                print(f"Failed to open JMP: {e}")
                print("Please open the following file in JMP manually:")
                print(path_to_open)
        else:
            print(f"JMP executable not found. Please open the following file in JMP:")
            print(path_to_open)
    else:
        print("\n\nNow install by opening the following file in JMP:")
        print(path_to_open)
        print("")