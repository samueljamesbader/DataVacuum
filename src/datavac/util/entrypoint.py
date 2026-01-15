import sys
from datavac.util.cli import CLIIndex


def util_cli_funcs():
    from datavac.util.caching import cli_clear_local_cache
    from datavac.util.rerun_data import cli_rerun_data
    from datavac.appserve.dvsecrets.ak_client_side import cli_refresh_access_key, cli_print_user, cli_invalidate_access_key
    return {
        'clear_cache': cli_clear_local_cache,
        'rerun_data (rd)': cli_rerun_data,
        'refresh_access_key (rak)': cli_refresh_access_key,
        'invalidate_access_key (iak)': cli_invalidate_access_key,
        'print_user (pu)': cli_print_user,
    }

def cli_launch_apps():
    from datavac.appserve.panel_serve import launch
    launch()

def entrypoint_datavac_cli():
    from datavac.database.db_cli import DB_CLI
    from datavac.config.contexts import CONTEXT_CLI
    from datavac.appserve.dvsecrets.vaults.vault import VAULT_CLI
    from datavac.jmp.compile_addin import cli_compile_jmp_addin
    UTIL_CLI= CLIIndex(util_cli_funcs)
    CLIIndex({
        'database (db)': DB_CLI,
        'context (cn)': CONTEXT_CLI,
        'util (ut)': UTIL_CLI,
        'vault (v)': VAULT_CLI,
        'compile_jmp (cj)': cli_compile_jmp_addin,
        'launch_apps (la)': cli_launch_apps,
    })(*sys.argv)


if __name__ == '__main__':
    entrypoint_datavac_cli()