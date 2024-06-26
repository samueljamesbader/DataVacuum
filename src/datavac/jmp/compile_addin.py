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

from urllib3.exceptions import InsecureRequestWarning

from datavac import logger
from datavac.util.conf import CONFIG

jmp_folder=Path(os.environ['DATAVACUUM_CACHE_DIR'])/"JMP"
jmp_folder.mkdir(exist_ok=True)

def copy_in_file(filename,addin_folder,addin_id):
    with irsc.as_file(irsc.files('datavac.jmp')) as jmpfiles:
        with open(jmpfiles/filename,'r') as f1:
            with open(addin_folder/filename,'w') as f2:
                f2.write(f1.read().replace("%ADDINID%",addin_id))

def make_db_connect(addin_folder,addin_id, env_values):
    connection_info=dict([[s.strip() for s in x.split("=")] for x in
                          env_values['DATAVACUUM_DBSTRING'].split(";")])
    #print(env_values)
    content= \
        f"""
            Names Default To Here(1);
            ADDIN_HOME = Get Path Variable("ADDIN_HOME({addin_id})");
            file_path=Convert File Path((ADDIN_HOME || "rootcertfile.crt"),windows);
            While(Pat Match(file_path,"\\","%BACKSLASH%"),1);
            While(Pat Match(file_path,"%BACKSLASH%","\\\\"),1);
            conn_str = 
                "ODBC:DRIVER={{PostgreSQL Unicode(x64)}};
                DATABASE={connection_info['Database']};
                SERVER={connection_info['Server']};
                PORT={connection_info['Port']};
                UID={connection_info['Uid']};
                PWD={connection_info['Password']};
                {'SSLmode=verify-full;' if connection_info['Server']!='localhost' else ''}
                ReadOnly=0;Protocol=7.4;FakeOidIndex=0;ShowOidColumn=0;RowVersioning=0;
                ShowSystemTables=0;Fetch=100;UnknownSizes=0;MaxVarcharSize=255;MaxLongVarcharSize=8190;
                Debug=0;CommLog=0;UseDeclareFetch=0;TextAsLongVarchar=1;UnknownsAsLongVarchar=0;BoolsAsChar=1;
                Parse=0;LFConversion=1;UpdatableCursors=1;TrueIsMinus1=0;BI=0;ByteaAsLongVarBinary=1;
                UseServerSidePrepare=1;LowerCaseIdentifier=0;
                {f'pqopt={{sslrootcert=%ROOTFILE%}};' if 'DATAVACUUM_DB_SSLROOTCERT' in env_values else '' }
                D6=-101;OptionalErrors=0;FetchRefcursors=0;XaOpt=1;";
            Pat Match(conn_str,"%ROOTFILE%",file_path);
            New SQL Query(
                Connection( conn_str),
                QueryName( "test_query" ),
                CustomSQL("Select * from information_schema.tables;"),
                PostQueryScript( "Close(Data Table(\!"test_query\!"), No Save);" )
                ) << Run;
                Print("Should now be connected to DB for {addin_id}");
            """
    with open(addin_folder/'dbconn.jsl','w') as f2:
        f2.write(content)

    if (rootcertfile:=env_values.get('DATAVACUUM_DB_SSLROOTCERT',None)):
        if Path(rootcertfile).exists():
            shutil.copy(rootcertfile,addin_folder/"rootcertfile.crt")
        else:
            with open(addin_folder/'rootcertfile.crt','wb') as f2:
                with warnings.catch_warnings():
                    logger.debug("Submitting an ssl-unverified request to download SSLROOTCERT.")
                    warnings.filterwarnings(category=InsecureRequestWarning,action='ignore')
                    f2.write(requests.get(env_values['DATAVACUUM_DB_SSLROOTCERT_DOWNLOAD'],verify=False).content)

def cli_compile_jmp_addin(*args):
    parser=argparse.ArgumentParser(description='Makes a .jmpaddin')
    parser.add_argument('envname',nargs="*",help='Environments (ie *.env files) to use',default=[''])
    namespace=parser.parse_args(args)

    for env in namespace.envname:
        if env != '':
            dotenv_path=dotenv.find_dotenv(f".{env}.env")
            assert dotenv_path, f"Didn't find .{env}.env"
            env_values=dotenv.dotenv_values(dotenv_path)
        else:
            env_values = os.environ
        envname=(env if len(env) else "LOCAL")
        print(f"Using {envname}")

        addin_folder=jmp_folder/envname
        if addin_folder.exists():
            shutil.rmtree(addin_folder)
        addin_folder.mkdir(exist_ok=False)

        with open(addin_folder/"Addin.def",'w') as f:
            addin_id=f'datavacuum_helper.{envname.lower()}'
            f.writelines([
                f"id={addin_id}\n",
                f"name=DataVac_{envname.capitalize()}\n",
                "supportJmpSE=0\n",
                "addinVersion=1\n",
                "minJmpVersion=16",])

        commands=[
            {
                'name':'Connect to Wafermap',
                'tip':'Assign map role for current table',
                'filename':'ConnectToWafermap.jsl',
                'icon':None
            }
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
                comm_act.set("type","path")
                comm_act.text=fr"$ADDIN_HOME({addin_id})\{command['filename']}"
                comm.append((comm_tip:=gfg.Element("jm:tip")))
                comm_tip.text=command['tip']
                assert command['icon'] is None
                comm.append((comm_icon:=gfg.Element("jm:icon")))
                comm_icon.set('type','None')

                copy_in_file(command['filename'],addin_folder,addin_id=addin_id)

            tree=gfg.ElementTree(root)
            gfg.indent(tree,'  ')
            tree.write(f)

            copy_in_file('addinLoad.jsl',addin_folder,addin_id=addin_id)
            make_db_connect(addin_folder=addin_folder,addin_id=addin_id,env_values=env_values)

        shutil.make_archive(addin_folder/f"DataVac_{envname.capitalize()}",'zip', addin_folder)
        shutil.move(addin_folder/f"DataVac_{envname.capitalize()}.zip",addin_folder/f"DataVac_{envname.capitalize()}.jmpaddin")
