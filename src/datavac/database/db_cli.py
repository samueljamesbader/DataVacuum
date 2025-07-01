import argparse

from datavac.util.cli import CLIIndex

def cli_print_database(*args):
    parser= argparse.ArgumentParser(description="Print the current database connection information.")
    parser.add_argument('--disallow-escalation','-da', action='store_true',
                        help="Disallow escalation of the database connection mode.")
    namespace = parser.parse_args(args)

    from datavac.database.db_connect import DBConnectionMode
    from datavac.config.project_config import PCONF
    for mode in DBConnectionMode:
        from datavac.database.db_connect import get_db_connection_info, get_specific_db_connection_info
        try:
            if namespace.disallow_escalation:
                print(f"O {mode.name} connection: {get_specific_db_connection_info(mode)}\n")
            else:
                print(f"O {mode.name} connection: {get_db_connection_info(mode)}\n")
        except PermissionError as e:
            print(f"X Failed to get {mode.name} connection info: {e}\n")

#def cli_force_database(*args):
#    parser = argparse.ArgumentParser(description="Forces the database to adhere to the current data definition.")
#    namespace = parser.parse_args(args)
#    from datavac.database.db_modify import force_database
#    force_database()

def cli_create_all(*args):
    parser = argparse.ArgumentParser(description="Creates any database objects defined in the current data definition that don't already exist.")
    namespace = parser.parse_args(args)
    from datavac.database.db_create import create_all
    create_all()

def cli_ensure_clear_database(*args):
    parser = argparse.ArgumentParser(description="Forces the database to adhere to the current data definition.")
    namespace = parser.parse_args(args)
    from datavac.database.db_create import ensure_clear_database
    ensure_clear_database()


def cli_read_and_enter_data(*args): 
    parser = argparse.ArgumentParser(description="Read and enter data.")
    parser.add_argument('--trove', '-t', type=str, nargs='+', default=None,
                        help="Restrict to specified trove(s)")
    parser.add_argument('--meas-group', '-g', type=str, nargs='+', default=None,
                        help="Restrict to specified measurement group(s)")
    from datavac.config.data_definition import DDEF

    possible_slcs=list(set(sum((DDEF().ALL_SAMPLELOAD_COLNAMES(t) for t in DDEF().troves),[])))

    for col in possible_slcs:
        parser.add_argument(f'--{col.lower()}',type=str, nargs='+', default=None,
            help=f"Restrict to specified {col}(s)")
    
    namespace = parser.parse_args(args)
    print(namespace)

    sampleload_info = {}
    for col in possible_slcs:
        if getattr(namespace, col.lower()) is not None:
            sampleload_info[col] = getattr(namespace, col.lower())
            
    from datavac.database.db_upload_meas import read_and_enter_data
    read_and_enter_data(trove_names=namespace.trove, only_meas_groups=namespace.meas_group,
                        only_sampleload_info=sampleload_info)


DB_CLI = CLIIndex({
    'print': cli_print_database,
    #'force': cli_force_database,
    'create': cli_create_all,
    'clear': cli_ensure_clear_database,
    'upload (ud)': cli_read_and_enter_data,
    })