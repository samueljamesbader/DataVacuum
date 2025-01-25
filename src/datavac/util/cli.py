import sys
from functools import partial
from pathlib import Path
from typing import Optional, List, Callable

from datavac.util.util import import_modfunc

def cli_helper(cli_funcs) -> Callable[[Optional[List[str]]],None]:
    def do_cli(override_sysargs=None):
        args=override_sysargs if override_sysargs is not None else sys.argv.copy()
        try:
            sub_call=args[1]
            func_dotpath=cli_funcs[sub_call]
        except:
            print(f'Call like "{Path(args[0]).name} COMMAND" where COMMAND options are:')
            for name in sorted(cli_funcs):
                print(f"- {name}")
            exit()

        #sys.argv=[f'{args[0]} {args[1]}',*args[2:]]
        func_dotpath,nest=(func_dotpath[2:],True) if func_dotpath.startswith("->") else (func_dotpath,False)
        func=import_modfunc(func_dotpath)

        sys.argv=[(args[0]+' '+args[1]),*args[2:]]
        if nest: return func(override_sysargs=sys.argv)
        else: return func(*args[2:])
    return do_cli


datavac_cli_funcs={
    'check_layout_params_valid': 'datavac.io.layout_params:cli_layout_params_valid',
    'update_layout_params': 'datavac.io.database:cli_update_layout_params',
    'clear_database': 'datavac.io.database:cli_clear_database',
    'upload_data': 'datavac.io.database:cli_upload_data',
    'upload_all_data': 'datavac.io.database:cli_upload_all_data',
    'dump_extraction': 'datavac.io.database:cli_dump_extraction',
    'dump_measurement': 'datavac.io.database:cli_dump_measurement',
    'dump_analysis': 'datavac.io.database:cli_dump_analysis',
    'print_database': 'datavac.io.database:cli_print_database',
    'clear_reextract_list': 'datavac.io.database:cli_clear_reextract_list',
    'clear_reanalyze_list': 'datavac.io.database:cli_clear_reanalyze_list',
    'force_database': 'datavac.io.database:cli_force_database',
    'update_mask_info': 'datavac.io.database:cli_update_mask_info',
    'heal': 'datavac.io.database:cli_heal',
    'compile_jmp': 'datavac.jmp.compile_addin:cli_compile_jmp_addin',
    'launch_apps':  'datavac.appserve.panel_serve:launch',
    'base64encode':  'datavac.util.util:cli_base64encode',
    'generate_secret':  'datavac.util.util:cli_b64rand',
    'context':'->datavac.util.conf:cli_context',
}
datavac_cli_main=cli_helper(cli_funcs=datavac_cli_funcs)
