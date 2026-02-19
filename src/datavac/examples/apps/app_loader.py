import asyncio
import os
import subprocess
from pathlib import Path

import panel as pn
from panel.widgets import CrossSelector

from datavac.appserve.app import PanelApp

# TODO: This should be somewhere more central unless it's specifically load logs
LOG_DIR=Path(os.environ['DATAVACUUM_LOG_DIR'])
LOG_DIR.mkdir(parents=False,exist_ok=True)
READ_DIR=Path(os.environ['DATAVACUUM_READ_DIR'])

class AppLoader(PanelApp):

    def get_allowed_folders(self):
        return [f.name for f in READ_DIR.iterdir() if f.is_dir()]
    def get_page(self):

        folders=list(sorted(self.get_allowed_folders(),reverse=True))
        self.folder_preselector=CrossSelector(options=folders,width=670)
        self.incremental_checkbox = pn.widgets.Checkbox(name="Incremental upload (no changed metadata!)", value=True)
        self.load_button=pn.widgets.Button(name="Load")
        self.load_button.on_click(self.do_load)
        self.load_log_display=pn.widgets.TextAreaInput(
            width=1000,height=400,sizing_mode='fixed',
            max_length=100000
        )
        self.page.main.append(pn.Row(pn.Spacer(sizing_mode='stretch_both'), # type: ignore
                                pn.Column(
                                    pn.Row(
                                        pn.Spacer(sizing_mode='stretch_width'),
                                        pn.Column(
                                            self.folder_preselector,
                                            # place checkbox below the button, centered horizontally
                                            pn.Row(
                                                pn.HSpacer(),
                                                pn.Column(self.load_button, self.incremental_checkbox, align='center'),
                                                pn.HSpacer(),
                                                width=670,
                                            ),
                                        ),
                                        pn.Spacer(sizing_mode='stretch_width'),
                                    ),
                                    self.load_log_display,
                                    width=1000,
                                    sizing_mode='stretch_height'
                                ),
                                pn.Spacer(sizing_mode='stretch_both')))
        return self.page

    async def do_load(self,*args,**kwargs):
        self.load_button.disabled=True
        flags=[]
        try:
            # Add -i flag if incremental_checkbox is checked
            if self.incremental_checkbox.value: # type: ignore
                flags.append("-i")
            fargs=[a for folder in self.folder_preselector.value for a in ['--folder',folder]]
            if not len(fargs): return
            identifier="hmm"
            with open(LOG_DIR/(f'load_{identifier}.txt'),'w') as f1:
                sp=subprocess.Popen([f"datavac","db","upload-data",*fargs,*flags],stdout=f1,stderr=f1)
            with open(LOG_DIR/(f'load_{identifier}.txt'),'r') as f2:
                while sp.poll() is None:
                    self.load_log_display.value+=f2.read() # type: ignore
                    await asyncio.sleep(1)
                self.load_log_display.value+=f2.read() # type: ignore
        except Exception as e:
            raise e
        finally:
            self.load_button.disabled=False



if __name__=='__main__':
    import panel as pn
    pn.serve(AppLoader().get_page().servable())
