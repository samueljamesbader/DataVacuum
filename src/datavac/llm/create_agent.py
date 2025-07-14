from textwrap import dedent
from datavac.llm.llm_dbtools import describe_mg, list_mgs
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from datavac.config.project_config import PCONF

def get_agent():
    llm=PCONF().vault.get_llm_connection_factory()()

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

        For most questions, you will probably want to start with the `list_mgs` tool to get
        an overview of the available measurement groups.
        ''')

    memory = MemorySaver()
    tools = [list_mgs, describe_mg]
    agent = create_react_agent(llm, tools=tools, checkpointer=memory, prompt=prompt)
    return agent


