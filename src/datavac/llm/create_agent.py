from datavac.llm.llm_dbtools import describe_mg, list_mgs
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from datavac.config.project_config import PCONF

def get_agent():
    llm=PCONF().vault.get_llm_connection_factory()()

    memory = MemorySaver()
    tools = [list_mgs, describe_mg]
    agent = create_react_agent(llm, tools=tools, checkpointer=memory)
    return agent


