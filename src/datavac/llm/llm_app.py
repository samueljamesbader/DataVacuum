from typing import Any

from datavac.appserve.app import PanelApp
import panel as pn
from panel.template.base import BasicTemplate
from datavac.llm.panel_langchain import PanelCallbackHandler

class HideToolsCallbackHandler(PanelCallbackHandler):
    """
    A custom callback handler that hides the tools in the chat interface.
    """
    def __init__(self, instance: pn.chat.ChatInterface):
        super().__init__(instance)
        self.instance = instance

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *args, **kwargs): pass
    def on_tool_end(self, output: str, *args, **kwargs): pass

class LLMApp(PanelApp):
    """
    A Panel application that provides an interface to interact with a language model.
    It allows users to ask questions about measurement groups and receive responses.
    """
    def __init__(self, **params):
        super().__init__(**params)
        from datavac.config.project_config import PCONF
        assert (llmm:=PCONF().llm_manager) is not None, "LLM Manager is not configured in the project configuration."
        self.agent = llmm.get_agent()

    def get_page(self) -> BasicTemplate:
        self.instance = pn.chat.ChatInterface(callback=self.callback,callback_exception='verbose')
        self.page.main.append( # type: ignore
            pn.Column(
                pn.pane.Markdown("## Measurement Group Information"),
                self.instance,
                sizing_mode='stretch_width',
            )
        )
        return self.page

    async def callback(self, contents, user, instance):
        if not hasattr(self, 'callback_handler'):
            self.callback_handler = HideToolsCallbackHandler(instance)
        await self.agent.ainvoke({'messages':[{'role':'user','content':contents}]},
                                 config=dict(callbacks=[self.callback_handler],thread_id=0)) # type: ignore

    
if 'bokeh' in  __name__:
    app=LLMApp()
    app.get_page().servable()