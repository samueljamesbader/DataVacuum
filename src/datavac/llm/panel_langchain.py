""" This module provides a PanelCallbackHandler for Langchain.
It is based off of the old panel.chat.langchain module with minor changes.
See https://github.com/holoviz/panel/releases/tag/v1.8.2

We much appreciate the work of the Panel project in creating this module.
Because it has since been rendered non-functional by langchain updates
and was deprecated by the Panel project in v1.8.0, DataVacuum is in-sourcing it.
(This note does not imply any endorsement by Holoviz/Panel of DataVacuum.)

Below is the license (BSD 3-Clause) under which the original Panel code was released.
(The below should not be interpreted as a license for the DataVacuum project.)

---------------------------------------------------------
---------------------------------------------------------
---------------------------------------------------------
Copyright (c) 2018, HoloViz team (holoviz.org).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

 * Neither the name of the copyright holder nor the names of any
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------
---------------------------------------------------------
---------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.callbacks.base import BaseCallbackHandler
from panel.chat.message import DEFAULT_AVATARS
from panel.layout import Accordion
from panel.util.warnings import deprecated

if TYPE_CHECKING:
    from typing import Any
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs.llm_result import LLMResult
    from panel.chat.feed import ChatFeed
    from panel.chat.interface import ChatInterface


class PanelCallbackHandler(BaseCallbackHandler):
    """
    The Langchain `PanelCallbackHandler` itself is not a widget or pane, but is useful for rendering
    and streaming the *chain of thought* from Langchain Tools, Agents, and Chains
    as `ChatMessage` objects.

    Reference: https://panel.holoviz.org/reference/chat/PanelCallbackHandler.html

    :Example:

    >>> chat_interface = pn.widgets.ChatInterface(callback=callback, callback_user="Langchain")
    >>> callback_handler = pn.widgets.langchain.PanelCallbackHandler(instance=chat_interface)
    >>> llm = ChatOpenAI(streaming=True, callbacks=[callback_handler])
    >>> chain = ConversationChain(llm=llm)

    """

    def __init__(
        self,
        instance: ChatFeed | ChatInterface,
        user: str = "LangChain",
        avatar: str = DEFAULT_AVATARS["langchain"],
    ):
        self.instance = instance
        self._message = None
        self._active_user = user
        self._active_avatar = avatar
        self._disabled_state = self.instance.disabled
        self._is_streaming = None

        self._input_user = user  # original user
        self._input_avatar = avatar

    def _update_active(self, avatar: str, label: str):
        """
        Prevent duplicate labels from being appended to the same user.
        """
        # not a typo; Langchain passes a string :/
        if label == "None":
            return

        self._active_avatar = avatar
        if f"- {label}" not in self._active_user:
            self._active_user = f"{self._active_user} - {label}"

    def _reset_active(self):
        self._active_user = self._input_user
        self._active_avatar = self._input_avatar
        self._message = None

    def _on_start(self, serialized, kwargs):
        model = kwargs.get("invocation_params", {}).get("model_name", "")
        self._is_streaming = serialized.get("kwargs", {}).get("streaming")
        messages = self.instance.objects
        if messages[-1].user != self._active_user:
            self._message = None
        if self._active_user and model not in self._active_user:
            self._active_user = f"{self._active_user} ({model})"

    def _stream(self, message: str):
        if message:
            return self.instance.stream(
                message,
                user=self._active_user,
                avatar=self._active_avatar,
                message=self._message,
            )
        return self._message

    def on_llm_start(self, serialized: dict[str, Any], *args, **kwargs):
        self._on_start(serialized, kwargs)
        return super().on_llm_start(serialized, *args, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._message = self._stream(token)
        return super().on_llm_new_token(token, **kwargs)

    def on_llm_end(self, response: LLMResult, *args, **kwargs):
        if not self._is_streaming:
            # on_llm_new_token does not get called if not streaming
            self._stream(response.generations[0][0].text)

        self._reset_active()
        return super().on_llm_end(response, *args, **kwargs)

    def on_llm_error(self, error: Exception | KeyboardInterrupt, *args, **kwargs):
        return super().on_llm_error(error, *args, **kwargs)

    def on_agent_action(self, action: AgentAction, *args, **kwargs: Any) -> Any:
        return super().on_agent_action(action, *args, **kwargs)

    def on_agent_finish(self, finish: AgentFinish, *args, **kwargs: Any) -> Any:
        return super().on_agent_finish(finish, *args, **kwargs)

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *args, **kwargs
    ):
        self._update_active(DEFAULT_AVATARS["tool"], serialized["name"])
        self._stream(f"Tool input: {input_str}")
        return super().on_tool_start(serialized, input_str, *args, **kwargs)

    def on_tool_end(self, output: str, *args, **kwargs):
        self._stream(output)
        self._reset_active()
        return super().on_tool_end(output, *args, **kwargs)

    def on_tool_error(
        self, error: Exception | KeyboardInterrupt, *args, **kwargs
    ):
        return super().on_tool_error(error, *args, **kwargs)

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], *args, **kwargs
    ):
        self._disabled_state = self.instance.disabled
        self.instance.disabled = True
        return super().on_chain_start(serialized, inputs, *args, **kwargs)

    def on_chain_end(self, outputs: dict[str, Any], *args, **kwargs):
        self.instance.disabled = self._disabled_state
        return super().on_chain_end(outputs, *args, **kwargs)

    def on_retriever_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: Any
    ) -> Any:
        """Run when Retriever errors."""
        return super().on_retriever_error(error, **kwargs)

    def on_retriever_end(self, documents, **kwargs: Any) -> Any:
        """Run when Retriever ends running."""
        objects = [(f"Document {index}", document.page_content) for index, document in enumerate(documents)]
        message = Accordion(*objects, sizing_mode="stretch_width", margin=(10,13,10,5))
        self.instance.send(
            message,
            user="LangChain (retriever)",
            avatar=DEFAULT_AVATARS["retriever"],
            respond=False,
        )
        return super().on_retriever_end(documents=documents, **kwargs)

    def on_text(self, text: str, **kwargs: Any):
        """Run when text is received."""
        return super().on_text(text, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list,
        **kwargs: Any
    ) -> None:
        """
        To prevent the inherited class from raising
        NotImplementedError, will not call super() here.
        """
        self._on_start(serialized, kwargs)
