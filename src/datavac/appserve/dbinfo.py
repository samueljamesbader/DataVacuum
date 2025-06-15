from datavac.io.database import PostgreSQLDatabase
from tornado.web import RequestHandler

from datavac.util.conf import CONFIG

#from datavac.appserve.ad_auth import with_validated_access_key
#class DBTableDescribe(RequestHandler):
#    @with_validated_access_key
#    def post(self, validated: dict[str,str]):
#        self.set_header('Content-Type', 'application/json')
#        self.write({'User': validated['User'],})

def cdict_to_str(cdict:dict[str,str]) -> str:
    """Convert a dictionary of columns and pandas dtypes (as specified in project.yaml) to a string representation with sqlalchemy dtypes."""
    return '\n'.join([f"- \"{k}\": {PostgreSQLDatabase.pd_to_sql_types[v].__visit_name__}" for k, v in cdict.items()])

def describe_mgoa(mgoa):
    response = ""
    if mgoa in CONFIG['measurement_groups']:
        response+= f"""'{mgoa}' is a group of measurements"""
        if (tdesc:=CONFIG['measurement_groups'][mgoa].get('description')):
            response+= f""" with the following description: "{tdesc}".  """
        else:
            response+= f""".  """
        response+= f"""It is represented by the view "{mgoa}" in the schema "jmp" of the postgresql database.  """
        response+= f"""This explainer will give the names of the columns in this view, as well as their sqlalchemy datatypes.  """
        response+= f"""First, the table has some columns describing the samples measured:\n"""
        response+= cdict_to_str(dict(**{k:'string' for k in CONFIG.ALL_MATLOAD_COLUMNS}))
        response+= f"""\nwhere {CONFIG.FULL_MATNAME_COL} is the fully specified unique sample name.  """
        response+= f"""\nThis table view has some columns describing the collection of the data:\n"""
        response+= cdict_to_str(CONFIG['measurement_groups'][mgoa]['meas_columns'])
        response+= f"""\nIt also has the following columns of parameters extracted from the measurements:\n"""
        response+= cdict_to_str(CONFIG['measurement_groups'][mgoa]['analysis_columns'])
    return response

def list_tables():
    """List all measurement groups and their descriptions."""
    response = "The following measurement groups are available:\n"
    for mgoa in CONFIG['measurement_groups']:
        response += f"- {mgoa}: {CONFIG['measurement_groups'][mgoa].get('description', 'No description available')}\n"
    return response
