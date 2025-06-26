import datetime
from io import StringIO
import os
from importlib import resources as irsc
from pathlib import Path
from typing import Optional, TextIO, cast
from tornado.web import RequestHandler
from yaml import safe_load

def get_setup_contents(yesno_code:str='yes', output_buffer: Optional[TextIO]=None):
    """Writes or returns a setup script for the the active DataVacuum deployment.

    Args:
        yesno_code: 'yes' for yes-code setup, 'no' for no-code setup.
        output_buffer: If provided, writes the script to this buffer.
            Otherwise, returns the script as a string.
    """
    assert yesno_code in ['yes','no'], "yesno_code must be 'yes' or 'no'"

    # If output_buffer is not provided, create a StringIO buffer
    sio= output_buffer or StringIO()
    
    # Get deployment name and URL from environment variables
    depname=os.environ['DATAVACUUM_DEPLOYMENT_NAME']
    depurl =os.environ['DATAVACUUM_DEPLOYMENT_URI']

    # Get the package that contains the deployment setup script from server_index.yaml
    # TODO: centralize this in CONFIG
    index_yaml_file=Path(os.environ['DATAVACUUM_CONFIG_DIR'])/"server_index.yaml"
    with open(index_yaml_file, 'r') as f:
        f=f.read()
        for k,v in os.environ.items():
            if 'DATAVAC' in k: f=f.replace(f"%{k}%",v)
        theyaml=safe_load(f)
    setup_script_package=theyaml['setup_script_package']

    # Write the setup script from the template
    sio.write(f":: {yesno_code.capitalize()}-code Setup file for '{depname}', downloaded {datetime.datetime.now()}\n")
    sio.write(f"set DATAVACUUM_DEPLOYMENT_URI={depurl}\n")
    sio.write(f"set DATAVACUUM_DEPLOYMENT_NAME={depname}\n")
    with irsc.as_file(irsc.files(setup_script_package)) as deployment_dir:
        with open(os.path.join(deployment_dir, f'{yesno_code}code_user_setup.bat'),'r') as f:
            for line in f:
                sio.write(line)
    
    # If the output_buffer was not provided, return the contents as a string
    if not output_buffer:
        return cast(StringIO,sio).getvalue()


class NoCodeSetupDownload(RequestHandler):
    def get(self):
        depname=os.environ['DATAVACUUM_DEPLOYMENT_NAME']
        self.set_header('Content-Disposition', f'attachment; filename={depname}_nocodesetup.bat')
        get_setup_contents(yesno_code='no', output_buffer=self) # type: ignore

class YesCodeSetupDownload(RequestHandler):
    def get(self):
        depname=os.environ['DATAVACUUM_DEPLOYMENT_NAME']
        self.set_header('Content-Disposition', f'attachment; filename={depname}_yescodesetup.bat')
        get_setup_contents(yesno_code='yes', output_buffer=self) # type: ignore
