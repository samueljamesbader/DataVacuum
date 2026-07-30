from typing import Any

from datavac.appserve.app import PanelApp
import panel as pn
from panel.template.base import BasicTemplate
from datavac.util.dvlogging import logger


class LLMApp(PanelApp):
    """
    A Panel application that provides an interface to interact with a language model.
    It allows users to ask questions about measurement groups and receive responses.
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.instance=None
        self.agent=None

    def get_page(self) -> BasicTemplate:
        from datavac.config.project_config import PCONF
        assert (llmm:=PCONF().llm_manager) is not None, "LLM Manager is not configured in the project configuration."

        def get_instance():
            self.instance = pn.chat.ChatInterface(callback=self.callback,callback_exception='verbose')
            self.agent = llmm.get_agent()
            return self.instance
        self.page.main.append( # type: ignore
            pn.Column(
                pn.pane.Markdown("## Measurement Group Information"),
                get_instance,
                sizing_mode='stretch_width',
            )
        )
        return self.page

    async def callback(self, contents, user, instance):
        if not hasattr(self, 'callback_handler'):
            from datavac.llm.panel_langchain import HideToolsCallbackHandler
            self.callback_handler = HideToolsCallbackHandler(instance)
        logger.info(f"User {user} asked: {contents}")
        await self.agent.ainvoke({'messages':[{'role':'user','content':contents}]},
                                 config=dict(callbacks=[self.callback_handler],thread_id=0)) # type: ignore

    
if 'bokeh' in  __name__:
    app=LLMApp()
    app.get_page().servable()