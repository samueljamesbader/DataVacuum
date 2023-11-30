import sys

from datavac.util.util import import_modfunc

cli_funcs={
    'update_layout_params': 'datavac.io.database:cli_update_layout_params',
    'clear_database': 'datavac.io.database:cli_clear_database',
    'upload_data': 'datavac.io.database:cli_upload_data',
    'dump_extraction': 'datavac.io.database:cli_dump_extraction',
    'force_database': 'datavac.io.database:cli_force_database',
    'heal': 'datavac.io.database:cli_heal',
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