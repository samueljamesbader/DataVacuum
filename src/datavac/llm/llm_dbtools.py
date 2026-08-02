from __future__ import annotations
from typing import Callable

from langchain_core.tools import tool, BaseTool
from functools import wraps


def trycatchtool(func:Callable[..., str]) -> BaseTool:
    """Wraps a tool function to log or return exceptions instead of raising them."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        try: return func(*args, **kwargs)
        except Exception as e:
            import traceback
            from datavac.util.dvlogging import logger
            # log the whole traceback not just the exception message, return just the message
            logger.error(f"Exception in tool {func.__name__}: {e}\n{traceback.format_exc()}")
            import pdb; pdb.set_trace()
            return f"Error in tool {func.__name__}"
    return tool(wrapper)

@trycatchtool
def list_mgoas() -> str:
    """List all measurement groups and analyses and their descriptions."""
    from datavac.config.data_definition import DDEF
    from datavac.util.dvlogging import logger
    logger.info(f"Listing all measurement groups and analyses")
    response = "The following measurement groups are available:\n"
    for mg_name, mg in DDEF().measurement_groups.items():
        response += f"- (measurement group) {mg_name}: {mg.description}\n"
    response += "The following analyses are available:\n"
    for an_name, an in DDEF().higher_analyses.items():
        response += f"- (analysis) {an_name}: {an.description}\n"
    return response

@trycatchtool
def describe_mg(mg_name: str) -> str:
    """Describe the measurement group.
    
    Args:
        mg_name: The name of the measurement group to describe.
        
    Returns:
        A description of the measurement group, its tables, and their columns.
    """
    from datavac.util.dvlogging import logger
    from datavac.database.db_util import namewsq
    from datavac.config.data_definition import DDEF
    from datavac.database.db_create import create_meas_group_view
    from sqlalchemy.schema import CreateTable
    from datavac.database.db_connect import get_engine_ro
    logger.info(f"Describing measurement group: {mg_name}")
    try: mg = DDEF().measurement_groups[mg_name]
    except KeyError as e: return f"Measurement group '{mg_name}' not found."
    response = f"""
        |||'{mg_name}' is the name of a measurement group with the following description: "{mg.description}".
        |||The measurements are indexed in the table {namewsq(mg.dbtable('meas'))} with the following DDL:
        |||{CreateTable(mg.dbtable('meas')).compile(get_engine_ro(), compile_kwargs={"literal_binds": True})}
        |||The extracted parameters from the measurements are in {namewsq(mg.dbtable('extr'))} with the following DDL:
        |||{CreateTable(mg.dbtable('extr')).compile(get_engine_ro(), compile_kwargs={"literal_binds": True})}
        |||{'\n'.join((f'Further information is available in the table {namewsq(DDEF().subsample_references[ssr].dbtable())} with the following DDL:'+\
            str(CreateTable(DDEF().subsample_references[ssr].dbtable()).compile(get_engine_ro(), compile_kwargs={"literal_binds": True})))
             for ssr in mg.subsample_reference_names)}
        

        |||These various tables are already conveniently joined together in a view with the following DDL:
        |||{"\n"+create_meas_group_view(mg.name,conn=None,just_DDL_string=True)}
        |||
        |||When providing example code to a user, it can be easier (though not required) to query the above view
        |||rather than reconstructing specific joins of component tables, as it helps with readability.

        |||Here is more information about the columns discussed above:
        |||{'\n'.join([f'  - "{c.name}": {c.description}'
                    for c in (mg.meas_columns+\
                              [mg.available_extr_columns()[cn] for cn in mg.extr_column_names]+\
                                [c for ssr_name in mg.subsample_reference_names for c in [DDEF().subsample_references[ssr_name].key_column]\
                                                                                        +DDEF().subsample_references[ssr_name].info_columns])])}
        
        """.replace('        |||', '')
    return response

@trycatchtool
def describe_an(an_name: str) -> str:
    """Describe the analysis.

    Args:
        an_name: The name of the analysis to describe.

    Returns:
        A description of the analysis, its tables, and their columns.
    """
    from datavac.util.dvlogging import logger
    from datavac.database.db_util import namewsq
    from datavac.config.data_definition import DDEF
    from datavac.database.db_create import create_analysis_view
    from sqlalchemy.schema import CreateTable
    from datavac.database.db_connect import get_engine_ro
    logger.info(f"Describing analysis: {an_name}")
    try: an = DDEF().higher_analyses[an_name]
    except KeyError: return f"Analysis '{an_name}' not found."
    avail = an.available_analysis_columns()
    response = f"""
        |||'{an_name}' is the name of an analysis with the following description: "{an.description}".
        |||The analysis results are stored in the table {namewsq(an.dbtables('anls'))} with the following DDL:
        |||{CreateTable(an.dbtables('anls')).compile(get_engine_ro(), compile_kwargs={"literal_binds": True})}
        |||{'\n'.join((f'Further information is available in the table {namewsq(DDEF().subsample_references[ssr].dbtable())} with the following DDL:'+\
            str(CreateTable(DDEF().subsample_references[ssr].dbtable()).compile(get_engine_ro(), compile_kwargs={"literal_binds": True})))
             for ssr in an.subsample_reference_names)}


        |||These various tables are already conveniently joined together in a view with the following DDL:
        |||{"\n"+create_analysis_view(an.name,conn=None,just_DDL_string=True)}
        |||
        |||When providing example code to a user, it can be easier (though not required) to query the above view
        |||rather than reconstructing specific joins of component tables, as it helps with readability.

        |||Here is more information about the columns discussed above:
        |||{'\n'.join([f'  - "{c.name}": {c.description}'
                    for c in ([avail[cn] for cn in an.analysis_column_names]+\
                                [c for ssr_name in an.subsample_reference_names for c in [DDEF().subsample_references[ssr_name].key_column]\
                                                                                        +DDEF().subsample_references[ssr_name].info_columns])])}

        """.replace('        |||', '')
    return response

@trycatchtool
def readonly_sql(query: str) -> str:
    """Run a read-only SQL query against the DataVacuum database (postgresql) and return the results as a string.
    
    Warning: any use of '%' should be escaped as '%%'.
    """
    from datavac.database.db_util import read_only_sql
    from datavac.util.dvlogging import logger
    logger.info(f"Running read-only SQL query: {query}")
    try:
        df = read_only_sql(query)
    except Exception as e:
        return f"Error executing query: {str(e)}."
    return df.to_string(index=False)

if __name__ == "__main__":
    import os
    #os.environ['DATAVACUUM_CONTEXT'] = 'builtin:demo2'
    #print(describe_mg.invoke('IdVg'))
    print(describe_an.invoke('Gam Sort A1'))
    #print(readonly_sql.invoke('SELECT * FROM vac."Samples" LIMIT 5'))