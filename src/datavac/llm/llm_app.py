from datavac.appserve.app import PanelApp
from datavac.llm.create_agent import get_agent
import panel as pn
from panel.template.base import BasicTemplate

class HideToolsCallbackHandler(pn.chat.langchain.PanelCallbackHandler):
    """
    A custom callback handler that hides the tools in the chat interface.
    """
    def __init__(self, instance: pn.chat.ChatInterface):
        super().__init__(instance)
        self.instance = instance

    def on_tool_start(self, serialized, kwargs): pass
    def on_tool_end(self, serialized, kwargs): pass

class LLMApp(PanelApp):
    """
    A Panel application that provides an interface to interact with a language model.
    It allows users to ask questions about measurement groups and receive responses.
    """
    def __init__(self, **params):
        super().__init__(**params)
        self.agent = get_agent()

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
        callback_handler = HideToolsCallbackHandler(instance)
        await self.agent.ainvoke({'messages':[{'role':'user','content':contents}]}, dict(callbacks=[callback_handler],thread_id=0))#['messages'][-1].content
    
if 'bokeh' in  __name__:
    app=LLMApp()
    app.get_page().servable()