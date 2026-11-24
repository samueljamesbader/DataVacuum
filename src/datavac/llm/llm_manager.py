from __future__ import annotations
from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel
    from langchain_core.runnables import Runnable


class LLMManager:

    def get_base_chat_model(self) -> BaseChatModel:
        raise NotImplementedError("Subclass should implement")

    def get_agent(self) -> Runnable:
        llm=self.get_base_chat_model()

        from datavac.llm.llm_dbtools import DB_TOOLS_GUIDANCE, langchain_tools

        prompt=dedent('''
            You are an assistant that helps users interact with the DataVacuum database.

            {guidance}

            You can query the database to answer user questions.Or, if requested, you can also provide guidance,
            because you are an expert in Python and JMP JSL and SQL, who can instruct users on writing code
            to query the database and visualize results in their language of choice.
            You can also directly answer questions about the structure of the database, such as
            explaining what measurement groups or analyses are available, and what columns they contain.
            Users may appreciate seeing the queries you run so they learn more about the database when you
            answer their questions.

            If you have run a 2-3 tool calls without outputing any content for the user, it's a helpful to
            keep them updated on the steps you're taking so they don't think you're stuck.
            ''').replace('{guidance}', DB_TOOLS_GUIDANCE)

        from langgraph.checkpoint.memory import MemorySaver
        from langchain.agents import create_agent
        memory = MemorySaver()
        agent = create_agent(llm, tools=langchain_tools(), checkpointer=memory, system_prompt=prompt)
        return agent


