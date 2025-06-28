import textwrap
from datavac.database.db_util import namewsq
#from tornado.web import RequestHandler

from datavac.config.data_definition import DDEF
from langchain_core.tools import tool

#from datavac.appserve.ad_auth import with_validated_access_key
#class DBTableDescribe(RequestHandler):
#    @with_validated_access_key
#    def post(self, validated: dict[str,str]):
#        self.set_header('Content-Type', 'application/json')
#        self.write({'User': validated['User'],})

#def cdict_to_str(cdict:dict[str,str]) -> str:
#    """Convert a dictionary of columns and pandas dtypes (as specified in project.yaml) to a string representation with sqlalchemy dtypes."""
#    return '\n'.join([f"- \"{k}\": {PostgreSQLDatabase.pd_to_sql_types[v].__visit_name__}" for k, v in cdict.items()])

@tool
def list_mgs() -> str:
    """List all measurement groups and their descriptions."""
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

#def describe_mg(mg_name: str) -> str:
#    response = ""
#    mg = DDEF().measurement_groups[mg_name]
#    response+= textwrap.dedent(f"""
#        '{mg_name}' is the name of a measurement group with the following description: "{mg.description}".
#        The following tables store data for this measurement group:
#        - The information about the measurements is in {namewsq(mg.dbtable('meas'))}, with the following columns:
#        {'\n'.join([f'  - "{c.name}" ({c.sql_dtype}), with description: {c.description}'
#                    for c in mg.meas_columns])}
#        The {namewsq(mg.dbtable('meas'))} table has a two column primary key:
#          - "loadid", which is a foreign key to the {mg.trove().dbtables('loads')}
#          - "measid", which numbers the measurements within a load.
#        {'\n'.join(f'The {namewsq(mg.dbtable('meas'))} table also has a key "{DDEF().subsample_references[ssr].key_column.name}" which '
#             for ssr in mg.subsample_reference_names)}
#        - The extracted parameters from the measurements are in {namewsq(mg.dbtable('extr'))}, with the following columns:
#        {'\n'.join([f'  - "{c}" ({mg.available_extr_columns()[c].sql_dtype}), with description: {mg.available_extr_columns()[c].description}'
#                    for c in mg.extr_column_names])}
#        This table also has "loadid" and "measid" keys with which to join to the measurements.
#        - The information about the loads is in {namewsq(mg.trove().dbtables('loads'))}, with the following columns:
#            {'\n'.join([f'  - "{c.name}" ({c.sql_dtype}), with description: {c.description}'
#                        for c in mg.trove().load_info_columns])}
#        This table has a primary key "loadid" and a foreign key "sampleid" to the {namewsq(mg.trove().dbtables('samples'))} table.
#        - The information about the samples is in {namewsq(mg.trove().dbtables('samples'))}, with the following columns:
#            {'\n'.join([f'  - "{c.name}" ({c.sql_dtype}), with description: {c.description}'
#                        for c in DDEF().sample_info_columns])}
#
#        """)
#
#    response+= f"""It is represented by the view "{mgoa}" in the schema "jmp" of the postgresql database.  """
#    response+= f"""This explainer will give the names of the columns in this view, as well as their sqlalchemy datatypes.  """
#    response+= f"""First, the table has some columns describing the samples measured:\n"""
#    response+= cdict_to_str(dict(**{k:'string' for k in CONFIG.ALL_MATLOAD_COLUMNS}))
#    response+= f"""\nwhere {CONFIG.FULL_MATNAME_COL} is the fully specified unique sample name.  """
#    response+= f"""\nThis table view has some columns describing the collection of the data:\n"""
#    response+= cdict_to_str(CONFIG['measurement_groups'][mgoa]['meas_columns'])
#    response+= f"""\nIt also has the following columns of parameters extracted from the measurements:\n"""
#    response+= cdict_to_str(CONFIG['measurement_groups'][mgoa]['analysis_columns'])
#    return response

if __name__ == "__main__":
    import os
    os.environ['DATAVACUUM_CONTEXT'] = 'builtin:demo2'
    print(describe_mg('IdVg'))