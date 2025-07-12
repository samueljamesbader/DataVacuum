from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any, Callable, Union



class CLIIndex():

    _cli_func_lookup: dict[str, Callable]
    _cli_subcommands: list[str]

    def __init__(self, cli_funcs: dict[str,Callable]| Callable[[],dict[str,Callable]]):
        self._generate_cli_funcs= (cli_funcs if callable(cli_funcs) else lambda: cli_funcs)
        self._is_setup = False

    def _setup(self) -> None:
        if self._is_setup: return
        cli_funcs = self._generate_cli_funcs()
        self._cli_func_lookup={k.split("(")[0].strip():v for k,v in cli_funcs.items()}
        self._cli_subcommands = list(cli_funcs)
        for k,f in cli_funcs.items():
            if '(' in k:
                shorthand=k.split("(")[1].split(")")[0]
                assert shorthand not in self._cli_func_lookup
                self._cli_func_lookup[shorthand]=f
        self._is_setup = True

    def __call__(self, *args) -> None:
        self._setup()

        next_args = [' '.join(args[:2]),*args[2:]]
        try:
            sub_call=args[1]
            func=self._cli_func_lookup[sub_call]
        except:
            print(f'Call like "{Path(args[0]).name} COMMAND" where COMMAND options are:')
            for name in sorted(self._cli_subcommands):
                print(f"- {name}")
            print(f'Run "{Path(args[0]).name} COMMAND -h" for more info about any command')
            exit()
    
        # If the callable is a CLIIndex, we call it with all the arguments including the command name.
        if isinstance(func, CLIIndex):
            func(*next_args)

        # Otherwise, we supply all arguments except the command name, and but we temporarily store the
        # compounded command name in sys.argv[0] so that, eg, help messages from argparse can be printed correctly.
        else:
            try:
                stored_sys_argv=sys.argv.copy()
                sys.argv=next_args
                func(*next_args[1:])
            finally:
                sys.argv=stored_sys_argv

def util_cli_funcs():
    from datavac.util.caching import cli_clear_local_cache
    from datavac.util.rerun_data import cli_rerun_data
    return {
        'clear_cache': cli_clear_local_cache,
        'rerun_data (rd)': cli_rerun_data,
    }

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
    })(*sys.argv)
