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
    for trove in DDEF().troves:
        cli_expander = DDEF().troves[trove].cli_expander
        for kwarg_name, parser_args in cli_expander.items():
            parser.add_argument(*parser_args['name_or_flags'],
                                **{k: v for k, v in parser_args.items() if k != 'name_or_flags'}) # type: ignore

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
    trove_names= namespace.trove if namespace.trove is not None else list(DDEF().troves.keys())
    read_and_enter_data(trove_names=namespace.trove, only_meas_groups=namespace.meas_group,
                        only_sampleload_info=sampleload_info,
                        kwargs_by_trove={t: {k: getattr(namespace, pa['name_or_flags'][0].lstrip('--')) 
                                             for k, pa in DDEF().troves[t].cli_expander.items()}
                                         for t in trove_names})

def cli_update_layout_params(*args):
    from datavac.database.db_semidev import update_layout_params
    parser = argparse.ArgumentParser(description="Update layout parameters in the database.")
    parser.add_argument('--dump_extractions_and_analyses','-da', action='store_true',
                        help="Dump extractions and analyses after updating layout parameters.")
    namespace = parser.parse_args(args)
    update_layout_params(dump_extractions_and_analyses=namespace.dump_extractions_and_analyses)

def cli_update_measurement_groups(*args):
    from datavac.database.db_modify import update_measurement_group_tables
    parser = argparse.ArgumentParser(description="Update measurement groups in the database.")
    update_measurement_group_tables()

def cli_update_analysis_tables(*args):
    from datavac.database.db_modify import update_analysis_tables
    parser = argparse.ArgumentParser(description="Update analysis tables in the database.")
    update_analysis_tables()

    
def cli_sql(*args):
    from datavac.database.db_util import read_sql
    import traceback
    parser=argparse.ArgumentParser(description='Runs a SQL query')
    #parser.add_argument('query',help='SQL query to run')
    #parser.add_argument('-c','--commit',action='store_true',help='Commit the transaction')
    namespace=parser.parse_args(args)

    inp=input("Query?: ").strip()
    while inp!='q':
        if inp!='':
            try:
                result=read_sql(inp)#,commit=namespace.commit)
            except:
                # print stack trace
                traceback.print_exc()
            else:
                print(result)
        inp=input("Query?: ").strip()

def cli_run_new_analysis(*args):
    from datavac.database.db_modify import run_new_analysis
    parser = argparse.ArgumentParser(description="Runs a new analysis on all data in the database.")
    parser.add_argument('analysis_name', type=str, help='Name of the analysis to run')
    namespace = parser.parse_args(args)
    run_new_analysis(namespace.analysis_name)

def cli_heal(*args):
    from datavac.database.db_modify import heal
    parser = argparse.ArgumentParser(description="Heals the database by readding known-removed data")
    namespace = parser.parse_args(args)
    heal()

def cli_update_split_tables(*args):
    from datavac.database.db_semidev import update_split_tables
    parser = argparse.ArgumentParser(description="Update split tables in the database.")
    parser.add_argument('--specific-splits', '-s', type=str, nargs='*', default=None,
                        help="Specific splits to update, if None all splits are updated")
    parser.add_argument('--force', '-f', action='store_true',
                        help="Force update of split tables even if they are up-to-date")
    namespace = parser.parse_args(args)
    update_split_tables(specific_splits=namespace.specific_splits, force=namespace.force)

DB_CLI = CLIIndex({
    'print': cli_print_database,
    #'force': cli_force_database,
    'create': cli_create_all,
    'clear': cli_ensure_clear_database,
    'upload (ud)': cli_read_and_enter_data,
    'update-layout-params (ulp)': cli_update_layout_params,
    'update-measurement-groups (umg)': cli_update_measurement_groups,
    'update-analysis-tables (uat)': cli_update_analysis_tables,
    'update-split-tables (ust)': cli_update_split_tables,
    'heal (h)': cli_heal,
    'run-new-analysis (rna)': cli_run_new_analysis,
    'sql': cli_sql,
    })