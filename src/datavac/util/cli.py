import sys
from functools import partial
from pathlib import Path
from typing import Optional, List, Callable

from datavac.util.util import import_modfunc

def cli_helper(cli_funcs) -> Callable[[Optional[List[str]]],None]:
    cli_func_longhand={k.split("(")[0].strip():v for k,v in cli_funcs.items()}
    cli_func_abbrevs={k.split("(")[1].strip()[:-1]:k.split("(")[0].strip() for k in cli_funcs if '(' in k}
    assert all(k not in cli_func_longhand for k in cli_func_abbrevs), "Abbreviations must not overlap with longhand names"

    def do_cli(override_sysargs=None):
        args=override_sysargs if override_sysargs is not None else sys.argv.copy()
        try:
            sub_call=args[1]
            if sub_call in cli_func_abbrevs: sub_call=cli_func_abbrevs[sub_call]
            func_dotpath=cli_func_longhand[sub_call]
        except:
            print(f'Call like "{Path(args[0]).name} COMMAND" where COMMAND options are:')
            for name in sorted(cli_funcs):
                print(f"- {name}")
            print(f'Run "{Path(args[0]).name} COMMAND -h" for more info about any command')
            exit()

        func_dotpath,nest=(func_dotpath[2:],True) if func_dotpath.startswith("->") else (func_dotpath,False)
        func=import_modfunc(func_dotpath)

        initial_sys_argv=sys.argv.copy()
        try:
            sys.argv=[(args[0]+' '+args[1]),*args[2:]]
            if nest: return func(override_sysargs=sys.argv)
            else: return func(*args[2:])
        finally: sys.argv=initial_sys_argv
    return do_cli


datavac_cli_funcs={
    'compile_jmp (cj)': 'datavac.jmp.compile_addin:cli_compile_jmp_addin',
    'launch_apps (la)':  'datavac.appserve.panel_serve:launch',
    'context (cn)':'->datavac.util.conf:cli_context',
    'database (db)':'->datavac.io.database:cli_database',
    'layout_params (lp)':'->datavac.io.layout_params:cli_layout_params',
    'util (ut)':'->datavac.util.cli:cli_util',
}
datavac_cli_main=cli_helper(cli_funcs=datavac_cli_funcs)
cli_util=cli_helper(cli_funcs={
    'base64encode': 'datavac.util.util:cli_base64encode',
    'generate_secret': 'datavac.util.util:cli_b64rand',
    'ensure_valid_access_key (evak)': 'datavac.appserve.user_side:cli_ensure_valid_access_key',
})