import sys
from functools import partial

from datavac.util.util import import_modfunc

def cli_helper(cli_funcs,override_sysargs=None):
    args=override_sysargs if override_sysargs is not None else sys.argv
    try:
        sub_call=args[1]
        func_dotpath=cli_funcs[sub_call]
    except:
        print(f'Call like "{args[0]} COMMAND" where COMMAND options are:')
        for name in sorted(cli_funcs):
            print(f"- {name}")
        exit()

    #sys.argv=[f'{args[0]} {args[1]}',*args[2:]]
    func=import_modfunc(func_dotpath)

    return func(*args[2:])

datavac_cli_funcs={
    'update_layout_params': 'datavac.io.database:cli_update_layout_params',
    'clear_database': 'datavac.io.database:cli_clear_database',
    'upload_data': 'datavac.io.database:cli_upload_data',
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
    'launch_apps':  'datavac.appserve.panel_serve:launch'
}
datavac_cli_main=partial(cli_helper,cli_funcs=datavac_cli_funcs)
