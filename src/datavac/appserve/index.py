import panel as pn
from pathlib import Path
from panel.theme.material import MaterialDarkTheme
pn.extension('mathjax')

from .app import PanelApp
from slugify import slugify
from yaml import safe_load
import importlib

class Indexer():
    def __init__(self, index_yaml_file: Path):
        with open(index_yaml_file, 'r') as f:
            self.categorized_applications=safe_load(f)

        def get_app_function(appname,dotpath:str):
            def run_app():
                try:
                    module_path, class_name = dotpath.rsplit(':',maxsplit=1)
                    cls: PanelApp=getattr(importlib.import_module(module_path),class_name)
                except AttributeError as e:
                    raise Exception(f"Failed to load App {appname} from module {module_path}," \
                                    f" class {class_name}. Make sure the module and class are correct in index.yaml?")
                page=cls().get_page()
                #self.project.validate_page(page)
                return page

            return run_app

        self.slug_to_app = {slugify(k):get_app_function(k,v['app'])
                            for group,apps in self.categorized_applications.items()
                            for k,v in apps.items()}

        self.slug_to_role = {slugify(k):v['role']
                             for group,apps in self.categorized_applications.items()
                             for k,v in apps.items()}


class AppIndex(PanelApp):

    title = "DataVacuum Home"

    def get_page(self):
        template = self.page
        if pn.state.user_info:
            template.header.append(pn.pane.Markdown(f"## Welcome {pn.state.user_info['given_name']}"))
        main_row=pn.FlexBox()
        #main_row.append(pn.Spacer(sizing_mode='stretch_both'))
        for cat,apps in pn.state.cache['index'].categorized_applications.items():
            if cat=='': continue # Skip the Index itself
            c=pn.Card(title=cat)
            for appname,app in apps.items():
                c.append(button:=pn.widgets.Button(name=appname,button_style='outline'))
                button.js_on_click(code=f'window.open("{slugify(appname)}")')
            main_row.append(c)
        main_row.append(pn.Spacer(sizing_mode='stretch_both'))
        #main_row=pn.pane.Markdown("# System down for extended maintenance ~Sam",sizing_mode='stretch_both')
        template.main.append(main_row)
        template.sidebar_width=200
        template.sidebar.append(pn.pane.Markdown("# Status\nAll systems nominal"))
        #template.servable()
        return template


if __name__.startswith('bokeh_app'):
    AppIndex.get_page()
