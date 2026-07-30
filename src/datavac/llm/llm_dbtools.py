from langchain_core.tools import tool

@tool
def list_mgs() -> str:
    """List all measurement groups and their descriptions."""
    from datavac.config.data_definition import DDEF
    response = "The following measurement groups are available:\n"
    for mg_name, mg in DDEF().measurement_groups.items():
        response += f"- {mg_name}: {mg.description}\n"
    return response

@tool
def describe_mg(mg_name: str) -> str:
    """Describe the measurement group.
    
    Args:
        mg_name: The name of the measurement group to describe.
        
    Returns:
        A description of the measurement group, its tables, and their columns.
    """
    from datavac.database.db_util import namewsq
    from datavac.config.data_definition import DDEF
    from datavac.database.db_create import create_meas_group_view
    from sqlalchemy.schema import CreateTable
    from datavac.database.db_connect import get_engine_ro
    mg = DDEF().measurement_groups[mg_name]
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
        |||In general, it's best to use query the above view rather than the component tables, as it helps with readability.

        |||Here is more information about the columns discussed above:
        |||{'\n'.join([f'  - "{c.name}": {c.description}'
                    for c in (mg.meas_columns+\
                              [mg.available_extr_columns()[cn] for cn in mg.extr_column_names]+\
                                [c for ssr_name in mg.subsample_reference_names for c in [DDEF().subsample_references[ssr_name].key_column]\
                                                                                        +DDEF().subsample_references[ssr_name].info_columns])])}
        
        """.replace('        |||', '')
    return response

if __name__ == "__main__":
    import os
    os.environ['DATAVACUUM_CONTEXT'] = 'builtin:demo2'
    print(describe_mg.invoke('IdVg'))