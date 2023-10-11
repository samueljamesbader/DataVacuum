import panel as pn
pn.extension('mathjax')
from panel.template.base import BasicTemplate


class PanelApp:
    title: str = "DataVacuum"
    def __init__(self, title = None):
        page_kwargs={}
        if (theme:=pn.state.cache.get('theme',None)) is not None:
            page_kwargs['theme']=theme
        if title is not None:
            self.title=title
        self.page: BasicTemplate = pn.template.MaterialTemplate(title=self.title, **page_kwargs)
    def get_page(self) -> BasicTemplate:
        raise NotImplementedError("Subclass must implement get_page")


