import sys

from datavac.util.util import import_modfunc

cli_funcs={
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
}
def cli_main():
    try:
        sub_call=sys.argv[1]
        func_dotpath=cli_funcs[sub_call]
    except:
        print(f'Call like "datavac COMMAND" where COMMAND options are:')
        for name in cli_funcs:
            print(f"- {name}")
        exit()

    sys.argv=[f'{sys.argv[0]} {sys.argv[1]}',*sys.argv[2:]]
    func=import_modfunc(func_dotpath)

    return func(*sys.argv[1:])