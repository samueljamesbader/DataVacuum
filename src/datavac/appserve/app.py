import panel as pn
pn.config.reconnect = True
pn.config.notifications = True
pn.extension('mathjax') # type: ignore
from panel.template.base import BasicTemplate


class PanelApp:
    title: str = "DataVacuum"
    def __init__(self, title = None):
        from datavac.config.project_config import PCONF
        if title is not None: self.title=title
        self.page: BasicTemplate = pn.template.MaterialTemplate(title=self.title, theme=PCONF().server_config.get_theme())
    def get_page(self) -> BasicTemplate:
        raise NotImplementedError("Subclass must implement get_page")

#from dataclasses import dataclass
#from typing import Callable, Optional
#from tornado.web import RequestHandler
#@dataclass(kw_only=True)
#class PanelAPI(RequestHandler):
#    endpoint: str
#    api_func: Callable
#    require_access_key: bool = True
#    required_role: str|None
#
#    def __post_init__(self):
#        if self.required_role is not None:
#            assert self.require_access_key, "Cannot require role without requiring access key"
#
#    def run(self):
#        import inspect
#        kwargs={}
#        for p in inspect.signature(self.api_func).parameters.values():
#            if p.default is not inspect.Parameter.empty:
#                kwargs[p.name] = self.get_argument(p.name,str(p.default))
#            else:
#                try: kwargs[p.name] = self.get_argument(p.name)
#                except Exception as e:
#                    raise ValueError(f"Missing required argument '{p.name}' for API function {self.api_func.__name__}") from e
#        if self.require_access_key:
#            from datavac.appserve.dvsecrets.ak_server_side import with_role
#            with_role(self.required_role)(lambda self,**kwargs: self.api_func(**kwargs))(self,**kwargs)
#        else:
#            self.api_func(**kwargs)
#