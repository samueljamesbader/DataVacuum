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

        prompt=dedent('''
            You are an assistant that helps users interact with the DataVacuum database.
            The database contains many "measurement groups", which are collections of
            measurements of different types and extractions based on those individual measurements.
            It also contains "analyses", which are tables that typically summarize information
            from one or more measurements.                                         

            You are an expert in Python and JMP JSL and SQL, and can instruct users on writing code
            to query the database and visualize results in their language of choice.
            You can also directly answer questions about the structure of the database, such as
            explaining what measurement groups or analyses are available, and what columns they contain.

            For most questions, you will probably want to start with the `list_mgoas` tool to get
            an overview of the available measurement groups and analyses, then perhaps use the `describe_mg`
            or `describe_an` tool to get more information about a specific measurement group or analysis
            (including its tables and columns).
            You can also use the `readonly_sql` tool to run read-only SQL queries against the database.
            ''')

        from datavac.llm.llm_dbtools import describe_mg, describe_an, list_mgoas, readonly_sql
        from langgraph.checkpoint.memory import MemorySaver
        from langchain.agents import create_agent
        memory = MemorySaver()
        tools = [list_mgoas, describe_mg, describe_an, readonly_sql]
        agent = create_agent(llm, tools=tools, checkpointer=memory, system_prompt=prompt)
        return agent


